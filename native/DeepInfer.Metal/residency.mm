// Own Metal residency, not a second copy of model weights. No dummy GPU work.
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <vector>

class QResidency {
    id<MTLResidencySet> set;
    NSString *failure;
    std::mutex mutex;
    std::condition_variable changed;
    std::thread refresh;
    bool stopping=false, active=false;
    uint64_t bytes=0, budget;
    int interval;
public:
    QResidency(id<MTLDevice> device, int milliseconds, uint64_t limit)
        : budget(limit), interval(milliseconds) {
        auto descriptor=[MTLResidencySetDescriptor new];
        descriptor.label=@"Infer.NET model weights";
        NSError *error=nil;
        set=[device newResidencySetWithDescriptor:descriptor error:&error];
        failure=error.localizedDescription;
    }
    NSString *error() const { return failure ?: @"Cannot create residency set"; }
    bool valid() const { return set!=nil; }
    bool add(const std::vector<id<MTLBuffer>> &buffers) {
        std::lock_guard<std::mutex> lock(mutex);
        uint64_t size=0;for(auto b:buffers)size+=b.allocatedSize;
        if(size>budget-bytes)return false;
        for(auto b:buffers)[set addAllocation:b];
        bytes+=size;
        if(active){[set commit];[set requestResidency];}
        return true;
    }
    void remove(const std::vector<id<MTLBuffer>> &buffers) {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto b:buffers){[set removeAllocation:b];bytes-=b.allocatedSize;}
        if(active)[set commit];
    }
    void start() {
        std::lock_guard<std::mutex> lock(mutex);
        if(active)return;
        [set commit];[set requestResidency];active=true;
        fprintf(stderr,"RESIDENCY weightsBytes=%llu budgetBytes=%llu allocations=%lu refreshMs=%d\n",
            (unsigned long long)bytes,(unsigned long long)budget,(unsigned long)set.allocationCount,interval);
        if(interval>0)refresh=std::thread([this]{
            std::unique_lock<std::mutex> lock(mutex);
            while(!changed.wait_for(lock,std::chrono::milliseconds(interval),[this]{return stopping;})) {
                @autoreleasepool { if(bytes)[set requestResidency]; }
            }
        });
    }
    uint64_t allocated() { std::lock_guard<std::mutex> lock(mutex); return bytes; }
    ~QResidency() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        changed.notify_all();if(refresh.joinable())refresh.join();
        if(active)[set endResidency];
        [set removeAllAllocations];[set commit];
    }
};
// Graph layers copy QMatrix descriptors. Copies share a lease; only the last
// descriptor releases residency, so disposal/growth never retains dead buffers.
struct QResidencyLease {
    std::shared_ptr<QResidency> owner;
    std::vector<id<MTLBuffer>> buffers;
    QResidencyLease(std::shared_ptr<QResidency> o,std::vector<id<MTLBuffer>> b):owner(o),buffers(std::move(b)){}
    ~QResidencyLease(){owner->remove(buffers);}
};
