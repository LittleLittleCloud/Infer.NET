// Hardware-independent state-transaction checks for the graph snapshot C ABI (Issue #72, S0).
// A CPU stand-in writes deterministic recurrent/convolution/KV mutations through the same
// shared buffers the GPU kernels use; real-model continuation, logits and full-state identity are
// covered by the managed `speculative-check` harness. Run: bash scripts/check-metal-speculative-state.sh
#include "bridge.mm"
#include <vector>
#include <cstdio>
#include <cmath>

static int failures=0,checks=0;
#define CHECK(condition) do{checks++;if(!(condition)){failures++;fprintf(stderr,"FAILED line %d: %s\n",__LINE__,#condition);}}while(0)

static id<MTLBuffer> make_buffer(QContext *c,size_t bytes){return [c->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];}
static GModel *fabricate(QContext *c,uint32_t capacity,bool bf16=false){
    auto *g=new GModel{};g->compute=*c;g->capacity=capacity;g->kvBf16=bf16;g->epsilon=1e-6f;g->theta=1e7f;
    for(int i=0;i<40;i++){
        auto &l=g->layers[i];l.loaded=true;l.linear=(i+1)%4!=0;
        if(l.linear){l.state=make_buffer(c,size_t(32)*128*128*4);l.history=make_buffer(c,size_t(3)*8192*4);}
        else{l.keys=g_kv_buffer(g,size_t(capacity)*512);l.values=g_kv_buffer(g,size_t(capacity)*512);}
        l.m[GQ].rows=8192+i;l.m[GQ].columns=2048;l.m[GQ].bits=8;
    }
    g->logits=make_buffer(c,size_t(248320)*4);
    return g;
}
// Stand-in for one committed decode step. Touches every region of every recurrent tensor.
static void step(GModel *g,uint32_t token){
    uint32_t x=token*2654435761u+g->position*40503u+17u;
    for(int i=0;i<40;i++){
        auto &l=g->layers[i];
        if(l.linear){
            auto *s=static_cast<float*>(l.state.contents);size_t n=size_t(32)*128*128;
            for(size_t k=0;k<n;k+=61)s[k]=s[k]*0.75f+float((x+uint32_t(k)*31u+uint32_t(i))&1023u)*(1.0f/1024.0f);
            s[n-1]=s[n-1]*0.5f+float(x&255u);s[0]+=float((x>>3)&255u);
            auto *h=static_cast<float*>(l.history.contents);
            for(size_t k=0;k<3*8192;k+=7)h[k]=h[k]*0.5f+float((x^uint32_t(k)*17u)&511u)*(1.0f/512.0f);
        }else{
            for(int e=0;e<512;e++){
                float k=float((x+uint32_t(e)*13u+uint32_t(i))&2047u)*(1.0f/2048.0f),v=float((x^uint32_t(e)*7u)&2047u)*(1.0f/2048.0f);
                size_t at=size_t(g->position)*512+e;
                if(g->kvBf16){uint32_t kb,vb;memcpy(&kb,&k,4);memcpy(&vb,&v,4);static_cast<uint16_t*>(l.keys.contents)[at]=uint16_t(kb>>16);static_cast<uint16_t*>(l.values.contents)[at]=uint16_t(vb>>16);}
                else{static_cast<float*>(l.keys.contents)[at]=k;static_cast<float*>(l.values.contents)[at]=v;}
            }
        }
    }
    auto *logits=static_cast<float*>(g->logits.contents);for(int k=0;k<248320;k+=997)logits[k]=float((x+uint32_t(k))&255u);
    g->position++;
}
static std::vector<uint64_t> digest(GModel *g){std::vector<uint64_t> d(80);CHECK(di_graph_state_digest(g,d.data(),80)==0);return d;}
static bool same_state(GModel *a,GModel *b){
    if(a->position!=b->position)return false;
    if(digest(a)!=digest(b))return false;
    for(int layer=0;layer<40;layer++)for(int kind=0;kind<2;kind++){
        size_t bytes=a->layers[layer].linear?(kind?3*8192*4:size_t(32)*128*128*4):size_t(a->position)*512*(a->kvBf16?2:4);
        std::vector<unsigned char> x(bytes),y(bytes);
        if(di_graph_state_export(a,layer,kind,x.data(),int(bytes))||di_graph_state_export(b,layer,kind,y.data(),int(bytes))||memcmp(x.data(),y.data(),bytes))return false;
    }
    return true;
}
static uint32_t token_at(int i){return uint32_t((i*7919+13)%248320);}
static const uint64_t kRequest=0xabcdef01u;

// Speculative cycle against a single snapshot: consume the whole block, reject at `offset`
// (offset>=count means accept everything), restore, replay only the committed prefix.
static bool cycle(GModel *g,uint64_t snapshot,int base,int count,int offset,double *captureMs,double *restoreMs){
    char error[512]{};
    if(di_graph_snapshot_capture(g,snapshot,kRequest,captureMs,error,sizeof(error))!=0){fprintf(stderr,"capture: %s\n",error);return false;}
    for(int i=0;i<=count;i++)step(g,token_at(base+i)); // inputs: pending token plus proposals
    int accepted=offset<count?offset:count;
    if(accepted<count){
        if(di_graph_snapshot_restore(g,snapshot,kRequest,restoreMs,error,sizeof(error))!=0){fprintf(stderr,"restore: %s\n",error);return false;}
        for(int i=0;i<=accepted;i++)step(g,token_at(base+i));
    }
    return g->position==uint32_t(base+accepted+1);
}

int main(){@autoreleasepool {
    char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));
    if(!c){fprintf(stderr,"%s\n",error);return 1;}
    constexpr uint32_t capacity=96;

    // Every proposal offset for every block size, plus all-accepted, against a never-speculating reference.
    for(int block=1;block<=8;block++)for(int offset=0;offset<=block;offset++){
        auto *reference=fabricate(c,capacity),*candidate=fabricate(c,capacity);
        uint64_t snapshot=0;CHECK(di_graph_snapshot_create(candidate,&snapshot,error,sizeof(error))==0&&snapshot);
        for(int i=0;i<5;i++){step(reference,token_at(i));step(candidate,token_at(i));}
        double captureMs=-1,restoreMs=-1;
        CHECK(cycle(candidate,snapshot,5,block,offset,&captureMs,&restoreMs));
        int committed=(offset<block?offset:block)+1;
        for(int i=0;i<committed;i++)step(reference,token_at(5+i));
        CHECK(same_state(reference,candidate));
        CHECK(captureMs>=0&&(offset>=block?restoreMs==-1:restoreMs>=0));
        // A rejected block must not leak into the next committed token.
        step(reference,7);step(candidate,7);CHECK(same_state(reference,candidate));
        delete reference;delete candidate;
    }

    // Fused FP32 decode: restore rewinds only recurrent/convolution state and KV visibility. Fused attention keeps its
    // per-token partials in scratch that every forward rewrites and bounds by g->position, so the flag and scratch are
    // untouched by restore and nothing extra needs snapshotting.
    for(int offset=0;offset<=4;offset++){
        auto *reference=fabricate(c,capacity),*candidate=fabricate(c,capacity);
        reference->decodeFusedFp32=true;candidate->decodeFusedFp32=true;
        uint64_t snapshot=0;CHECK(di_graph_snapshot_create(candidate,&snapshot,error,sizeof(error))==0&&snapshot);
        for(int i=0;i<5;i++){step(reference,token_at(i));step(candidate,token_at(i));}
        CHECK(cycle(candidate,snapshot,5,4,offset,nullptr,nullptr));
        int committed=(offset<4?offset:4)+1;
        for(int i=0;i<committed;i++)step(reference,token_at(5+i));
        CHECK(same_state(reference,candidate)&&candidate->decodeFusedFp32);
        delete reference;delete candidate;
    }

    // Repeated rollback with mixed outcomes; compare after every cycle.
    {
        auto *reference=fabricate(c,capacity),*candidate=fabricate(c,capacity);
        uint64_t snapshot=0;CHECK(di_graph_snapshot_create(candidate,&snapshot,error,sizeof(error))==0);
        int position=0;
        for(int round=0;round<24&&position+10<int(capacity);round++){
            int block=1+round%8,offset=(round*5+3)%(block+1);
            CHECK(cycle(candidate,snapshot,position,block,offset,nullptr,nullptr));
            int committed=(offset<block?offset:block)+1;
            for(int i=0;i<committed;i++)step(reference,token_at(position+i));
            position+=committed;CHECK(same_state(reference,candidate));
        }
        delete reference;delete candidate;
    }

    // KV visibility: restore hides the rejected suffix, poison proves nothing reads it, replay overwrites it.
    {
        auto *g=fabricate(c,capacity);uint64_t snapshot=0;CHECK(di_graph_snapshot_create(g,&snapshot,error,sizeof(error))==0);
        for(int i=0;i<6;i++)step(g,token_at(i));
        auto before=digest(g);std::vector<unsigned char> keys(size_t(6)*512*4);
        CHECK(di_graph_state_export(g,3,0,keys.data(),int(keys.size()))==0);
        CHECK(di_graph_snapshot_capture(g,snapshot,kRequest,nullptr,error,sizeof(error))==0);
        for(int i=0;i<4;i++)step(g,token_at(50+i));
        CHECK(g->position==10);
        std::vector<unsigned char> longer(size_t(10)*512*4);CHECK(di_graph_state_export(g,3,0,longer.data(),int(longer.size()))==0);
        CHECK(di_graph_snapshot_restore(g,snapshot,kRequest,nullptr,error,sizeof(error))==0);
        CHECK(g->position==6&&digest(g)==before);
        CHECK(di_graph_state_export(g,3,0,longer.data(),int(longer.size()))!=0); // rejected suffix is no longer exportable
        CHECK(di_graph_state_export(g,3,0,keys.data(),int(keys.size()))==0);
        CHECK(di_graph_poison_kv_suffix(g,0)!=0&&di_graph_poison_kv_suffix(g,4097)!=0&&di_graph_poison_kv_suffix(nullptr,4)!=0);
        CHECK(di_graph_poison_kv_suffix(g,8)==0&&digest(g)==before); // visible state unchanged
        auto *k=static_cast<const uint32_t*>(g->layers[3].keys.contents),*v=static_cast<const uint32_t*>(g->layers[39].values.contents);
        for(int row=6;row<14;row++)CHECK(k[size_t(row)*512]==0x7fc00000u&&v[size_t(row)*512+511]==0x7fc00000u);
        CHECK(k[size_t(5)*512]!=0x7fc00000u);
        step(g,9);CHECK(k[size_t(6)*512]!=0x7fc00000u&&k[size_t(7)*512]==0x7fc00000u); // append overwrites exactly one row
        // Restoring to a position the graph has not reached is rejected.
        g->position=3;CHECK(di_graph_snapshot_restore(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0);
        delete g;
    }

    // Lifecycle: reset/stale, request identity, ownership, release, invalidation by earlier restore.
    {
        auto *a=fabricate(c,capacity),*b=fabricate(c,capacity);uint64_t sa=0,sb=0,extra=0;
        CHECK(di_graph_snapshot_create(a,&sa,error,sizeof(error))==0&&di_graph_snapshot_create(b,&sb,error,sizeof(error))==0&&sa!=sb);
        CHECK(di_graph_snapshot_capture(a,sa,kRequest,nullptr,error,sizeof(error))==0);
        uint64_t info[12]{};CHECK(di_graph_snapshot_info(a,sa,kRequest,info,12)==0&&info[0]==1&&info[1]==a->instanceId&&info[2]==a->epoch&&
            info[3]==kRequest&&info[4]==0&&info[5]==capacity&&info[6]==4&&info[7]==g_model_identity(a)&&info[8]==uint64_t(30)*(524288+24576)*4&&info[9]==1&&info[10]==1&&info[11]==sa);
        CHECK(di_graph_snapshot_info(a,sa,kRequest+1,info,12)==0&&info[9]==0); // wrong request is not restorable
        uint64_t uncaptured[12]{};CHECK(di_graph_snapshot_info(b,sb,kRequest,uncaptured,12)==0&&uncaptured[10]==0&&uncaptured[9]==0);
        CHECK(di_graph_snapshot_info(a,sa,kRequest,info,11)!=0&&di_graph_snapshot_info(a,sa,kRequest,nullptr,12)!=0&&di_graph_snapshot_info(a,sb,kRequest,info,12)!=0);
        CHECK(di_graph_snapshot_restore(b,sb,kRequest,nullptr,error,sizeof(error))!=0); // never captured
        CHECK(di_graph_snapshot_restore(b,sa,kRequest,nullptr,error,sizeof(error))!=0&&di_graph_snapshot_capture(b,sa,kRequest,nullptr,error,sizeof(error))!=0&&
              di_graph_snapshot_release(b,sa)!=0); // cross-graph handles are unknown
        CHECK(di_graph_snapshot_restore(a,sa,0,nullptr,error,sizeof(error))!=0&&di_graph_snapshot_restore(a,sa,kRequest+1,nullptr,error,sizeof(error))!=0);
        CHECK(di_graph_snapshot_restore(a,0,kRequest,nullptr,error,sizeof(error))!=0&&di_graph_snapshot_restore(a,0xdeadbeefcafeull,kRequest,nullptr,error,sizeof(error))!=0);
        CHECK(di_graph_snapshot_restore(nullptr,sa,kRequest,nullptr,error,sizeof(error))!=0&&di_graph_snapshot_capture(a,sa,0,nullptr,error,sizeof(error))!=0);
        for(int i=0;i<3;i++)step(a,token_at(i));
        CHECK(di_graph_snapshot_restore(a,sa,kRequest,nullptr,error,sizeof(error))==0&&a->position==0);
        // Reset makes every older snapshot stale, even if position and bytes would line up.
        for(int i=0;i<3;i++)step(a,token_at(i));
        CHECK(di_graph_reset(a)==0&&a->epoch==2);
        CHECK(di_graph_snapshot_restore(a,sa,kRequest,nullptr,error,sizeof(error))!=0&&strstr(error,"reset")!=nullptr);
        CHECK(di_graph_snapshot_info(a,sa,kRequest,info,12)==0&&info[9]==0);
        CHECK(di_graph_snapshot_capture(a,sa,kRequest+1,nullptr,error,sizeof(error))==0); // recapture reuses the owned storage
        // Restoring to an earlier snapshot invalidates snapshots from the discarded lineage.
        uint64_t early=0,late=0;CHECK(di_graph_snapshot_release(a,sa)==0&&di_graph_snapshot_create(a,&early,error,sizeof(error))==0&&di_graph_snapshot_create(a,&late,error,sizeof(error))==0);
        for(int i=0;i<4;i++)step(a,token_at(i));
        CHECK(di_graph_snapshot_capture(a,early,kRequest,nullptr,error,sizeof(error))==0);
        for(int i=0;i<3;i++)step(a,token_at(10+i));
        CHECK(di_graph_snapshot_capture(a,late,kRequest,nullptr,error,sizeof(error))==0);
        CHECK(di_graph_snapshot_restore(a,early,kRequest,nullptr,error,sizeof(error))==0);
        CHECK(di_graph_snapshot_restore(a,late,kRequest,nullptr,error,sizeof(error))!=0&&strstr(error,"invalidated")!=nullptr);
        CHECK(di_graph_snapshot_restore(a,early,kRequest,nullptr,error,sizeof(error))==0); // repeated rollback to the same snapshot
        CHECK(di_graph_snapshot_create(a,&extra,error,sizeof(error))!=0); // bounded live snapshots
        CHECK(di_graph_snapshot_release(a,late)==0&&di_graph_snapshot_release(a,late)!=0); // double release
        CHECK(di_graph_snapshot_restore(a,late,kRequest,nullptr,error,sizeof(error))!=0&&di_graph_snapshot_capture(a,late,kRequest,nullptr,error,sizeof(error))!=0);
        delete a;delete b; // graph destruction frees live snapshots
    }

    // Unsupported states, geometry changes and the prefix-import boundary.
    {
        auto *g=fabricate(c,capacity);uint64_t snapshot=0;CHECK(di_graph_snapshot_create(g,&snapshot,error,sizeof(error))==0);
        CHECK(di_graph_snapshot_capture(g,snapshot,kRequest,nullptr,error,sizeof(error))==0);
        g->kvBf16=true;CHECK(di_graph_snapshot_restore(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0&&strstr(error,"precision")!=nullptr);g->kvBf16=false;
        g->layers[5].m[GQ].rows++;CHECK(di_graph_snapshot_restore(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0&&strstr(error,"model")!=nullptr);g->layers[5].m[GQ].rows--;
        g->capacity=0;CHECK(di_graph_snapshot_restore(g,snapshot,kRequest,nullptr,error,sizeof(error))==0);g->capacity=capacity; // position 0 <= capacity
        g->splitActive=true;g->splitMissing=1;
        CHECK(di_graph_snapshot_capture(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0&&di_graph_snapshot_restore(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0);
        uint64_t extra=0;CHECK(di_graph_snapshot_create(g,&extra,error,sizeof(error))!=0&&di_graph_poison_kv_suffix(g,1)!=0);
        std::vector<unsigned char> bytes(size_t(32)*128*128*4);CHECK(di_graph_prefix_state(g,0,0,1,bytes.data(),int(bytes.size()),1)!=0); // import API still guarded, never used for restore
        g->splitActive=false;g->splitMissing=0;g->forcedRoutes=make_buffer(c,64);
        CHECK(di_graph_snapshot_restore(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0&&di_graph_snapshot_capture(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0);
        g->forcedRoutes=nil;
        g->layers[7].loaded=false;CHECK(di_graph_snapshot_capture(g,snapshot,kRequest,nullptr,error,sizeof(error))!=0&&di_graph_state_digest(g,std::vector<uint64_t>(80).data(),80)!=0);g->layers[7].loaded=true;
        CHECK(di_graph_state_digest(g,nullptr,80)!=0&&di_graph_state_digest(g,std::vector<uint64_t>(80).data(),79)!=0);
        std::vector<unsigned char> tiny(4);CHECK(di_graph_state_export(g,0,0,tiny.data(),4)!=0&&di_graph_state_export(g,40,0,tiny.data(),4)!=0&&di_graph_state_export(g,0,2,tiny.data(),4)!=0);
        std::vector<float> logits(248320);CHECK(di_graph_last_logits(g,logits.data(),248320)==0&&di_graph_last_logits(g,logits.data(),10)!=0);
        // No snapshot memory exists for a graph that never snapshots.
        auto *quiet=fabricate(c,capacity);CHECK(quiet->snapshots==nullptr&&di_graph_snapshot_release(quiet,1)!=0&&di_graph_snapshot_restore(quiet,1,kRequest,nullptr,error,sizeof(error))!=0&&quiet->snapshots==nullptr);
        delete quiet;delete g;
    }

    // BF16 KV uses the same logical-position contract.
    {
        auto *reference=fabricate(c,capacity,true),*candidate=fabricate(c,capacity,true);
        uint64_t snapshot=0;CHECK(di_graph_snapshot_create(candidate,&snapshot,error,sizeof(error))==0);
        for(int i=0;i<3;i++){step(reference,token_at(i));step(candidate,token_at(i));}
        CHECK(cycle(candidate,snapshot,3,6,2,nullptr,nullptr));for(int i=0;i<3;i++)step(reference,token_at(3+i));
        CHECK(same_state(reference,candidate));delete reference;delete candidate;
    }
    delete c;
    if(failures){fprintf(stderr,"%d of %d speculative-state checks failed\n",failures,checks);return 1;}
    printf("PASS: %d speculative-state transaction checks (all block sizes 1-8 x every rejection offset and all-accepted, repeated/mixed rollback, KV visibility with NaN poison, reset/stale/request/ownership/release/invalidation lifecycle, split/diagnostic/geometry rejection, BF16 contract)\n",checks);
    return 0;
}}
