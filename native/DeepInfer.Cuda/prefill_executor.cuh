#pragma once
// Experimental layer-major executor, same 32-token wire chunk and FP32 state.
struct ChunkProfile {
    Model *model;std::vector<cudaEvent_t> events;std::vector<int> phases;
    explicit ChunkProfile(Model *g):model(g) {if(g->profileEnabled)mark(-1);}
    ~ChunkProfile(){for(auto event:events)cudaEventDestroy(event);}
    void mark(int phase){
        if(!model->profileEnabled)return;
        cudaEvent_t event;ck(cudaEventCreate(&event));events.push_back(event);phases.push_back(phase);ck(cudaEventRecord(event));
    }
    void complete(){
        if(events.empty())return;ck(cudaEventSynchronize(events.back()));
        for(size_t i=1;i<events.size();i++){float ms;ck(cudaEventElapsedTime(&ms,events[i-1],events[i]));model->profileTotals[phases[i]]+=ms;}
    }
};
static void prepare_chunk(Model *g) {
    auto &p=g->chunk;
    if(!p.prepared) {
#define CHUNK(name,n) p.name=g->alloc<float>(size_t(32)*(n))
        CHUNK(x,2048);CHUNK(norm,2048);CHUNK(qkv,8192);CHUNK(z,4096);CHUNK(a,32);CHUNK(b,32);
        CHUNK(conv,8192);CHUNK(attention,4096);CHUNK(k,512);CHUNK(v,512);CHUNK(q,4096);CHUNK(out,2048);
        CHUNK(router,256);CHUNK(prob,8);CHUNK(eg,4096);CHUNK(eu,4096);CHUNK(eo,16384);
        CHUNK(sg,512);CHUNK(su,512);CHUNK(so,2048);CHUNK(sr,1);p.ids=g->alloc<int>(32*8);p.tokens=g->alloc<int>(32);
#undef CHUNK
        p.prepared=true;
    }
    if(p.scoreCapacity<g->capacity) {
        auto *next=g->alloc<float>(size_t(32)*16*g->capacity);
        if(p.scores){auto old=std::find(g->memory.begin(),g->memory.end(),p.scores);if(old!=g->memory.end())g->memory.erase(old);ck(cudaFree(p.scores));}
        p.scores=next;p.scoreCapacity=g->capacity;
    }
}
static void chunk_mv(Model *g,Matrix m,const float *x,float *y,int count,const int *ids=nullptr,int experts=1,bool routed=false) {
    if(g->bf16Projections && !ids && m.rows>=256 && m.cols>=256)
        prefill_gemm_bf16<<<dim3((m.rows+63)/64,(count+31)/32),128>>>(m,x,y,count);
    else if(g->bf16AllProjections)
        chunk_gemv_bf16<<<dim3((m.rows+3)/4,experts,count),128>>>(m,x,y,ids,routed);
    else if(g->densePacked && !ids && count>=4 && m.rows>=256 && m.cols%256==0) {
        if(m.cols<=2048)chunk_dense_reuse<8,true><<<dim3((m.rows+3)/4,(count+7)/8),128>>>(m,x,y,count);
        else chunk_dense_reuse<4,true><<<dim3((m.rows+3)/4,(count+3)/4),128>>>(m,x,y,count);
    } else if(g->denseReuse && !ids && count>=4 && m.rows>=256)
        chunk_dense_reuse<4><<<dim3((m.rows+3)/4,(count+3)/4),128>>>(m,x,y,count);
    else if(g->expertPacked && m.cols%256==0)
        chunk_gemv_packed<<<dim3((m.rows+3)/4,experts,count),128>>>(m,x,y,ids,routed);
    else chunk_gemv<<<dim3((m.rows+3)/4,experts,count),128>>>(m,x,y,ids,routed);
}
static void chunk_stage_probe(const char *directory,const char *stage,const float *device,size_t floats) {
    std::vector<float> values(floats);
    ck(cudaMemcpy(values.data(),device,floats*sizeof(float),cudaMemcpyDeviceToHost));
    std::ofstream file(std::string(directory)+"/cuda-"+stage+".f32",std::ios::binary|std::ios::trunc);
    if(!file)throw std::runtime_error("Cannot open CUDA stage probe output");
    file.write(reinterpret_cast<const char*>(values.data()),std::streamsize(floats*sizeof(float)));
    if(!file)throw std::runtime_error("Cannot write CUDA stage probe output");
}
static void run_chunk(Model *g,const int *tokens,int count,float *hidden,float *trace) {
    if(g->profileEnabled&&trace)throw std::runtime_error("Phase profiling excludes trace readbacks");
    prepare_chunk(g);auto &p=g->chunk;int pos=g->position;ChunkProfile profile(g);
    const char *stageDir=trace&&pos==0?std::getenv("DEEPINFER_CUDA_STAGE_DIR"):nullptr;
    const bool stageProbe=stageDir&&*stageDir;
    auto probe=[&](const char *stage,const float *device,size_t floats,int layer){
        if(stageProbe&&layer==0)chunk_stage_probe(stageDir,stage,device,floats);
    };
    ck(cudaMemcpy(p.tokens,tokens,count*4,cudaMemcpyHostToDevice));
    chunk_embed<<<dim3(16,count),128>>>(g->embedding,p.tokens,p.x);
    probe("embedding",p.x,2048,0);
    for(int l=0;l<20;l++) {
        auto &v=g->layers[l];auto *m=v.m;
        chunk_norm<<<count,256>>>(p.x,nullptr,v.p,p.norm,g->eps,2048);
        probe("embedding-norm",p.norm,2048,l);
        const char *splitK=std::getenv("DEEPINFER_CUDA_QKV_SPLIT_K");
        if(g->bf16Projections&&splitK&&std::strcmp(splitK,"1")==0)
            prefill_gemm_bf16_split_k<<<dim3((m[0].rows+63)/64,(count+31)/32),128>>>(m[0],p.norm,p.qkv,count);
        else chunk_mv(g,m[0],p.norm,p.qkv,count);
        if((l+1)%4) {
            chunk_mv(g,m[3],p.norm,p.z,count);chunk_mv(g,m[4],p.norm,p.a,count);chunk_mv(g,m[5],p.norm,p.b,count);
            probe("qkv",p.qkv,8192,l);
            profile.mark(0);
            chunk_linear<<<dim3(64,count),128>>>(p.qkv,v.history,v.p,p.conv,p.a,p.b);
            prefill_delta_sequence<<<2048,64>>>(v.state,p.conv,p.a,p.b,p.attention,count);
            chunk_history<<<64,128>>>(p.qkv,v.history,count);
            chunk_delta_norm<<<dim3(32,count),128>>>(p.attention,p.z,v.p+36928,g->eps);
            probe("delta-norm",p.attention,4096,l);
        } else {
            chunk_mv(g,m[1],p.norm,p.k,count);chunk_mv(g,m[2],p.norm,p.v,count);
            profile.mark(0);
            if(g->kvBf16) {
                chunk_head<<<dim3(18,count),256>>>(p.qkv,p.k,p.v,v.p,p.q,(__nv_bfloat16*)v.keys,(__nv_bfloat16*)v.values,pos,g->eps,g->theta);
                chunk_scores<<<dim3((pos+count+3)/4,16,count),128>>>(p.q,(__nv_bfloat16*)v.keys,p.scores,pos,g->capacity);
                chunk_attend<<<dim3(16,count),256>>>(p.scores,(__nv_bfloat16*)v.values,p.qkv,p.attention,pos,g->capacity);
            } else {
                chunk_head<<<dim3(18,count),256>>>(p.qkv,p.k,p.v,v.p,p.q,(float*)v.keys,(float*)v.values,pos,g->eps,g->theta);
                chunk_scores<<<dim3((pos+count+3)/4,16,count),128>>>(p.q,(float*)v.keys,p.scores,pos,g->capacity);
                chunk_attend<<<dim3(16,count),256>>>(p.scores,(float*)v.values,p.qkv,p.attention,pos,g->capacity);
            }
        }
        profile.mark((l+1)%4 ? 1 : 2);
        chunk_mv(g,m[6],p.attention,p.out,count);
        probe("output-projection",p.out,2048,l);
        chunk_norm<<<count,256>>>(p.x,p.out,v.p+2048,p.norm,g->eps,2048);
        probe("residual-norm",p.norm,2048,l);
        profile.mark(3);
        if(g->bf16AllProjections&&std::getenv("DEEPINFER_CUDA_ROUTER_FP32")&&
           std::strcmp(std::getenv("DEEPINFER_CUDA_ROUTER_FP32"),"1")==0)
            chunk_gemv<<<dim3((m[7].rows+3)/4,1,count),128>>>(m[7],p.norm,p.router,nullptr,false);
        else chunk_mv(g,m[7],p.norm,p.router,count);
        const bool sharedFp32=g->bf16AllProjections&&std::getenv("DEEPINFER_CUDA_SHARED_FP32")&&
            std::strcmp(std::getenv("DEEPINFER_CUDA_SHARED_FP32"),"1")==0;
        if(sharedFp32){
            chunk_gemv<<<dim3((m[11].rows+3)/4,1,count),128>>>(m[11],p.norm,p.sg,nullptr,false);
            chunk_gemv<<<dim3((m[12].rows+3)/4,1,count),128>>>(m[12],p.norm,p.su,nullptr,false);
            chunk_gemv<<<dim3((m[14].rows+3)/4,1,count),128>>>(m[14],p.norm,p.sr,nullptr,false);
        }else{
            chunk_mv(g,m[11],p.norm,p.sg,count);chunk_mv(g,m[12],p.norm,p.su,count);chunk_mv(g,m[14],p.norm,p.sr,count);
        }
        probe("router",p.router,256,l);
        chunk_top8<<<count,32>>>(p.router,p.ids,p.prob);
        int probeLayer=0,probeTokenA=49,probeTokenB=91;
        const char *routeProbe=std::getenv("DEEPINFER_CUDA_ROUTE_PROBE");
        if(routeProbe&&std::sscanf(routeProbe,"%d:%d",&probeLayer,&probeTokenA)==2)probeTokenB=-1;
        if(trace && routeProbe && l==probeLayer)for(int token:{probeTokenA,probeTokenB})if(token>=0&&pos<=token&&token<pos+count){
            int ids[8];float probabilities[8];int local=token-pos;
            ck(cudaMemcpy(ids,p.ids+local*8,sizeof(ids),cudaMemcpyDeviceToHost));
            ck(cudaMemcpy(probabilities,p.prob+local*8,sizeof(probabilities),cudaMemcpyDeviceToHost));
            fprintf(stderr,"CUDA_ROUTE token=%d layer=%d",token,l);
            for(int i=0;i<8;i++)fprintf(stderr," %d:%.9g",ids[i],probabilities[i]);
            fputc('\n',stderr);
            if(probeTokenB<0){
                float norm[2048],logits[256];
                ck(cudaMemcpy(norm,p.norm+local*2048,sizeof(norm),cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(logits,p.router+local*256,sizeof(logits),cudaMemcpyDeviceToHost));
                fprintf(stderr,"CUDA_ROUTE_NORM token=%d layer=%d",token,l);
                for(float value:norm)fprintf(stderr," %.9g",value);
                fputc('\n',stderr);
                fprintf(stderr,"CUDA_ROUTE_LOGITS token=%d layer=%d",token,l);
                for(float value:logits)fprintf(stderr," %.9g",value);
                fputc('\n',stderr);
            }
        }
        activation<<<count*4,128>>>(p.sg,p.su,count*512);
        probe("shared-gate",p.sg,512,l);
        profile.mark(4);
        chunk_mv(g,m[8],p.norm,p.eg,count,p.ids,8);chunk_mv(g,m[9],p.norm,p.eu,count,p.ids,8);
        activation<<<count*32,128>>>(p.eg,p.eu,count*4096);
        probe("expert-gate",p.eg,4096,l);
        profile.mark(5);
        chunk_mv(g,m[10],p.eg,p.eo,count,p.ids,8,true);chunk_mv(g,m[13],p.sg,p.so,count);
        probe("expert-down",p.eo,16384,l);probe("shared-down",p.so,2048,l);
        profile.mark(6);
        chunk_merge<<<dim3(16,count),128>>>(p.x,p.eo,p.so,p.sr,p.prob);
        probe("merge",p.x,2048,l);
        profile.mark(7);
        if(trace)for(int t=0;t<count;t++)ck(cudaMemcpy(trace+(t*20+l)*2048,p.x+t*2048,2048*4,cudaMemcpyDeviceToHost));
    }
    ck(cudaGetLastError());ck(cudaMemcpy(hidden,p.x,size_t(count)*2048*4,cudaMemcpyDeviceToHost));g->position+=count;profile.mark(8);profile.complete();
}
