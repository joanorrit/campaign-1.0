CAMPAIGN_METRIC_LIB = euclideansquared
CAMPAIGN_METRIC_NAME=CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED

# bin_PROGRAMS = hello kcentersGPU kmeansGPU hierGPU  somGPU kcentersCPU kmeansCPU somCPU hierCPU
bin_PROGRAMS =  hierarchicalCPU \
		hierarchicalGPU \
		kcentersCPU \
		kcentersGPU \
		kmeansCPU \
		kmeansGPU \
		kmedoidsCPU \
		kmedoidsGPU \
		kpsmeansGPU \
		rmsdGPU \
		somCPU \
		somGPU


hierarchicalCPU_SOURCES = hierarchicalCPU.cc 
hierarchicalCPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
hierarchicalCPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
hierarchicalCPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

hierarchicalGPU_SOURCES = hierarchicalGPU.cc
hierarchicalGPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
hierarchicalGPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
hierarchicalGPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

kcentersCPU_SOURCES = kcentersCPU.cc 
kcentersCPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
kcentersCPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
kcentersCPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

kcentersGPU_SOURCES = kcentersGPU.cc
kcentersGPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
kcentersGPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
kcentersGPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

kmeansCPU_SOURCES = kmeansCPU.cc 
kmeansCPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
kmeansCPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
kmeansCPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

kmeansGPU_SOURCES = kmeansGPU.cc
kmeansGPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
kmeansGPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
kmeansGPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

kmedoidsCPU_SOURCES = kmedoidsCPU.cc 
kmedoidsCPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
kmedoidsCPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
kmedoidsCPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

kmedoidsGPU_SOURCES = kmedoidsGPU.cc
kmedoidsGPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
kmedoidsGPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
kmedoidsGPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

kpsmeansGPU_SOURCES = kpsmeansGPU.cc
kpsmeansGPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
kpsmeansGPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
kpsmeansGPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

rmsdGPU_SOURCES = rmsdGPU.cc
rmsdGPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
rmsdGPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
rmsdGPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

somCPU_SOURCES = somCPU.cc 
somCPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
somCPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
somCPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

somGPU_SOURCES = somGPU.cc
somGPU_LDFLAGS = -L/usr/local/cuda/lib -lcuda -lcudart
somGPU_LDADD = -lcampaign-$(CAMPAIGN_METRIC_LIB)-1.0
somGPU_CPPFLAGS = -I/usr/local/cuda/include -I/Developer/GPU\ Computing/C/common/inc -D$(CAMPAIGN_METRIC_NAME)

