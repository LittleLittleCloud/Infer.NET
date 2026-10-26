// Default-off graph state transactions for the speculative-state diagnostic (minimal reuse of reviewed S0 afdb36fd logic for Issue #76; no prefix-import API).
//
// A snapshot copies the FP32 DeltaNet state and convolution history of the 30 linear
// layers and records the logical position that bounds KV visibility for the 10
// attention layers. KV is append-only: restoring lowers g->position, and every
// attention kernel reads only keys/values at indices <= the current position, so a
// rejected suffix is inaccessible without copying or clearing it. Handles are opaque
// 64-bit identifiers resolved through the owning graph; no caller-held pointer is
// ever dereferenced. This file never uses the prefix-handoff import API.
//
// Threading: like the rest of the graph ABI, calls must be serialized by the caller.
// Snapshot memory is allocated only by di_graph_snapshot_create.
#include <chrono>
#include <memory>
#include <unordered_map>

namespace {
constexpr size_t kStateFloats=size_t(32)*128*128, kHistoryFloats=size_t(3)*8192;
constexpr size_t kLinearFloats=kStateFloats+kHistoryFloats;
constexpr int kInfoValues=12;
std::atomic<uint64_t> g_next_snapshot_handle{1};
}
struct GSnapshot {
    uint64_t handle=0,graph=0,epoch=0,request=0,model=0;
    uint32_t position=0,capacity=0,kvBytes=0;
    bool captured=false,valid=false;
    std::unique_ptr<float[]> data;
    size_t floats=0;
};
struct GSnapshotStore {
    std::unordered_map<uint64_t,std::unique_ptr<GSnapshot>> items;
};
static void g_snapshot_destroy(GSnapshotStore *s){delete s;}

static uint64_t g_mix(uint64_t h,uint64_t v){h^=v;h*=0x100000001b3ull;return h^(h>>29);}
// Identity of the loaded model geometry and weight buffers; a snapshot never restores into a different model.
static uint64_t g_model_identity(const GModel *g){
    uint64_t h=0xcbf29ce484222325ull;
    uint32_t e,t;memcpy(&e,&g->epsilon,4);memcpy(&t,&g->theta,4);h=g_mix(g_mix(h,e),t);
    h=g_mix(h,(uint64_t)(uintptr_t)(__bridge void*)g->embedding.packed);
    h=g_mix(h,(uint64_t)(uintptr_t)(__bridge void*)g->classifier.packed);
    for(int i=0;i<40;i++){
        auto &l=g->layers[i];h=g_mix(h,l.loaded?(l.linear?2:3):1);
        for(auto &m:l.m)h=g_mix(g_mix(g_mix(g_mix(h,m.rows),m.columns),m.bits),(uint64_t)(uintptr_t)(__bridge void*)m.packed);
    }
    return h;
}
static bool g_all_layers_loaded(const GModel *g){for(auto &l:g->layers)if(!l.loaded)return false;return true;}
// Transactions require a plain, non-diagnostic, non-split request state.
static const char *g_transaction_blocker(const GModel *g){
    if(g->blockPending||g->blockFaulted)return "Pending/faulted native target block";
    if(!g_all_layers_loaded(g))return "Graph layers are not fully registered";
    if(g->splitActive||g->splitMissing||g->splitTarget)return "State transactions are unavailable during split prefix handoff";
    if(g->forcedRoutes)return "State transactions are unavailable with diagnostic routing";
    return nullptr;
}
static GSnapshot *g_snapshot_find(GModel *g,uint64_t handle){
    if(!g->snapshots)return nullptr;
    auto it=g->snapshots->items.find(handle);return it==g->snapshots->items.end()?nullptr:it->second.get();
}
static const char *g_snapshot_blocker(GModel *g,const GSnapshot *s,uint64_t request,bool restoring){
    if(!s->captured)return "Snapshot has not captured state";
    if(s->graph!=g->instanceId)return "Snapshot belongs to a different graph";
    if(!s->valid)return "Snapshot was invalidated by a rollback to an earlier position";
    if(s->epoch!=g->epoch)return "Snapshot is stale: graph was reset after capture";
    if(s->request!=request)return "Snapshot belongs to a different request";
    if(auto *blocker=g_transaction_blocker(g))return blocker;
    if(s->kvBytes!=(g->kvBf16?2u:4u))return "Snapshot KV precision differs from the graph";
    if(s->model!=g_model_identity(g))return "Snapshot model identity differs from the graph";
    if(s->position>g->capacity)return "Snapshot position exceeds graph capacity";
    if(restoring&&g->position<s->position)return "Cannot restore to a position the graph has not reached";
    return nullptr;
}
static double g_since_ms(std::chrono::steady_clock::time_point start){
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}

extern "C" {
// Allocates reusable snapshot storage (~63 MiB for 30 FP32 recurrent layers). Contains no state until captured.
int di_graph_snapshot_create(void *graph,uint64_t *handle,char *error,int capacity){
    @autoreleasepool {
        if(!graph||!handle){error_text(error,capacity,@"Invalid snapshot create arguments");return -1;}
        auto *g=static_cast<GModel*>(graph);*handle=0;
        if(auto *blocker=g_transaction_blocker(g)){error_text(error,capacity,[NSString stringWithUTF8String:blocker]);return -1;}
        if(!g->snapshots)g->snapshots=new(std::nothrow) GSnapshotStore;
        if(!g->snapshots){error_text(error,capacity,@"Cannot allocate snapshot store");return -1;}
        if(g->snapshots->items.size()>=2){error_text(error,capacity,@"At most two snapshots may be live per graph");return -1;}
        auto item=std::make_unique<GSnapshot>();
        for(auto &l:g->layers)if(l.linear)item->floats+=kLinearFloats;
        item->data.reset(new(std::nothrow) float[item->floats]);
        if(!item->data){error_text(error,capacity,@"Cannot allocate snapshot storage");return -1;}
        item->handle=g_next_snapshot_handle.fetch_add(1);item->graph=g->instanceId;
        *handle=item->handle;g->snapshots->items[item->handle]=std::move(item);return 0;
    }
}
// Captures the committed state at the current position. The caller supplies a request identity.
int di_graph_snapshot_capture(void *graph,uint64_t handle,uint64_t request,double *ms,char *error,int capacity){
    @autoreleasepool {
        if(!graph||!request){error_text(error,capacity,@"Invalid snapshot capture arguments");return -1;}
        auto *g=static_cast<GModel*>(graph);auto *s=g_snapshot_find(g,handle);
        if(!s){error_text(error,capacity,@"Unknown, released or foreign snapshot handle");return -1;}
        if(auto *blocker=g_transaction_blocker(g)){error_text(error,capacity,[NSString stringWithUTF8String:blocker]);return -1;}
        if(g->position>g->capacity){error_text(error,capacity,@"Graph position exceeds capacity");return -1;}
        auto start=std::chrono::steady_clock::now();
        float *cursor=s->data.get();
        for(auto &l:g->layers)if(l.linear){
            memcpy(cursor,l.state.contents,kStateFloats*4);cursor+=kStateFloats;
            memcpy(cursor,l.history.contents,kHistoryFloats*4);cursor+=kHistoryFloats;
        }
        s->epoch=g->epoch;s->request=request;s->position=g->position;s->capacity=g->capacity;s->kvBytes=g->kvBf16?2u:4u;
        s->model=g_model_identity(g);s->captured=true;s->valid=true;
        if(ms)*ms=g_since_ms(start);return 0;
    }
}
// Restores recurrent/convolution state and lowers the logical position. Positions above it become unreadable.
int di_graph_snapshot_restore(void *graph,uint64_t handle,uint64_t request,double *ms,char *error,int capacity){
    @autoreleasepool {
        if(!graph||!request){error_text(error,capacity,@"Invalid snapshot restore arguments");return -1;}
        auto *g=static_cast<GModel*>(graph);auto *s=g_snapshot_find(g,handle);
        if(!s){error_text(error,capacity,@"Unknown, released or foreign snapshot handle");return -1;}
        if(auto *blocker=g_snapshot_blocker(g,s,request,true)){error_text(error,capacity,[NSString stringWithUTF8String:blocker]);return -1;}
        auto start=std::chrono::steady_clock::now();
        const float *cursor=s->data.get();
        for(auto &l:g->layers)if(l.linear){
            memcpy(l.state.contents,cursor,kStateFloats*4);cursor+=kStateFloats;
            memcpy(l.history.contents,cursor,kHistoryFloats*4);cursor+=kHistoryFloats;
        }
        g->position=s->position;g->tapValidRows=0;
        // Other snapshots taken after this position describe a lineage that no longer exists.
        for(auto &entry:g->snapshots->items)if(entry.second.get()!=s&&entry.second->position>s->position)entry.second->valid=false;
        if(ms)*ms=g_since_ms(start);return 0;
    }
}
// Read-only identity report; no lifecycle precondition other than ownership. Version 1 is 12 values.
// [0]=version [1]=graph instance [2]=epoch [3]=request [4]=position [5]=capacity [6]=KV element bytes
// [7]=model identity [8]=bytes [9]=restorable now [10]=captured [11]=handle
int di_graph_snapshot_info(void *graph,uint64_t handle,uint64_t request,uint64_t *values,int count){
    if(!graph||!values||count!=kInfoValues)return -1;
    auto *g=static_cast<GModel*>(graph);auto *s=g_snapshot_find(g,handle);if(!s)return -1;
    const uint64_t info[kInfoValues]={1,s->graph,s->epoch,s->request,s->position,s->capacity,s->kvBytes,s->model,
        uint64_t(s->floats)*4,uint64_t(g_snapshot_blocker(g,s,request,false)==nullptr),uint64_t(s->captured),s->handle};
    memcpy(values,info,sizeof(info));return 0;
}
int di_graph_snapshot_release(void *graph,uint64_t handle){
    if(!graph)return -1;
    auto *g=static_cast<GModel*>(graph);
    if(!g->snapshots)return -1;
    return g->snapshots->items.erase(handle)==1?0:-1;
}

// ---- Read-only diagnostics (never imports state; independent of the prefix-handoff API) ----
static uint64_t g_digest(const void *data,size_t bytes){
    uint64_t h=g_mix(0xcbf29ce484222325ull,bytes);const uint64_t *w=static_cast<const uint64_t*>(data);
    for(size_t i=0;i<bytes/8;i++)h=g_mix(h,w[i]);
    return h;
}
static bool g_layer_bytes(const GModel *g,int layer,int kind,size_t *bytes){
    auto &l=g->layers[layer];if(!l.loaded)return false;
    *bytes=l.linear?(kind?kHistoryFloats*4:kStateFloats*4):size_t(g->position)*512*(g->kvBf16?2:4);return true;
}
static id<MTLBuffer> g_layer_buffer(const GModel *g,int layer,int kind){
    auto &l=g->layers[layer];return l.linear?(kind?l.history:l.state):(kind?l.values:l.keys);
}
// 80 digests: [layer*2+kind]. Recurrent layers hash state/history; attention layers hash only the visible KV rows.
int di_graph_state_digest(void *graph,uint64_t *digests,int count){
    if(!graph||!digests||count!=80)return -1;
    auto *g=static_cast<GModel*>(graph);if(g->blockPending||g->blockFaulted)return -1;
    if(!g_all_layers_loaded(g))return -1;
    for(int layer=0;layer<40;layer++)for(int kind=0;kind<2;kind++){
        size_t bytes=0;if(!g_layer_bytes(g,layer,kind,&bytes))return -1;
        digests[layer*2+kind]=g_digest(g_layer_buffer(g,layer,kind).contents,bytes);
    }
    return 0;
}
// Number of positions the target has consumed (the visible KV length). Verification hook for committed-prefix replay.
int di_graph_position(void *graph,int *position){
    if(!graph||!position)return -1;*position=int(static_cast<GModel*>(graph)->position);return 0;
}
// Exact copy of one layer tensor for first-divergence evidence. Layout matches di_graph_prefix_state.
int di_graph_state_export(void *graph,int layer,int kind,void *data,int bytes){
    if(!graph||!data||layer<0||layer>=40||kind<0||kind>1)return -1;
    auto *g=static_cast<GModel*>(graph);size_t expected=0;
    if(!g_layer_bytes(g,layer,kind,&expected)||expected!=size_t(bytes))return -1;
    memcpy(data,g_layer_buffer(g,layer,kind).contents,expected);return 0;
}
// Copy of the most recent classifier output. The caller owns validity (which position produced it).
int di_graph_last_logits(void *graph,float *logits,int count){
    if(!graph||!logits||count!=248320)return -1;
    auto *g=static_cast<GModel*>(graph);memcpy(logits,g->logits.contents,size_t(248320)*4);return 0;
}
// Verification hook: fills KV rows [position, position+rows) of every attention layer with quiet NaNs.
// Correct decode never reads them, and a read becomes a rejected non-finite classifier result.
int di_graph_poison_kv_suffix(void *graph,int rows){
    if(!graph||rows<1||rows>4096)return -1;
    auto *g=static_cast<GModel*>(graph);if(g_transaction_blocker(g))return -1;
    uint32_t last=std::min<uint32_t>(g->capacity,g->position+uint32_t(rows));
    for(auto &l:g->layers)if(!l.linear)for(uint32_t row=g->position;row<last;row++)for(int e=0;e<512;e++){
        if(g->kvBf16){static_cast<uint16_t*>(l.keys.contents)[size_t(row)*512+e]=0x7fc0;static_cast<uint16_t*>(l.values.contents)[size_t(row)*512+e]=0x7fc0;}
        else{static_cast<uint32_t*>(l.keys.contents)[size_t(row)*512+e]=0x7fc00000u;static_cast<uint32_t*>(l.values.contents)[size_t(row)*512+e]=0x7fc00000u;}
    }
    return 0;
}
}
