#include "bridge.mm"
#include <cassert>
int main(){@autoreleasepool {
    setenv("DEEPINFER_METAL_RESIDENCY","1",1);
    setenv("DEEPINFER_METAL_RESIDENCY_REFRESH_MS","20",1);
    char error[2048]{};
    auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));
    if(!c){fprintf(stderr,"%s\n",error);return 1;}
    auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,1,64,64,8,64,error,sizeof(error)));
    if(!m){fprintf(stderr,"%s\n",error);return 1;}
    uint64_t bytes=c->residency->allocated();assert(bytes>0);
    {QMatrix graphCopy=*m;
        di_quant_residency_prepare(c);
        di_quant_residency_prepare(c); // Idempotent: never starts a second heartbeat.
        di_quant_matrix_destroy(m);
        assert(c->residency->allocated()==bytes);
        std::this_thread::sleep_for(std::chrono::milliseconds(65));
    }
    assert(c->residency->allocated()==0);
    // Reject an oversized lease without partially registering its buffers.
    auto tiny=std::make_shared<QResidency>(c->device,0,1);
    auto buffer=[c->device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    assert(tiny->valid()&&!tiny->add({buffer})&&tiny->allocated()==0);
    tiny.reset();di_quant_destroy(c);
    setenv("DEEPINFER_METAL_RESIDENCY","0",1);
    c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));
    assert(c&&!c->residency);di_quant_residency_prepare(c);di_quant_destroy(c);
    puts("PASS residency: registration, copy lifetime, last-release cleanup, repeated prepare, budget rejection, disabled mode, heartbeat shutdown");
}}
