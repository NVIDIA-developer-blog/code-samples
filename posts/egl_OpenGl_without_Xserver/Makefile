OBJECTS = $(filter-out %.c,$(notdir $(patsubst %.cc,%.o,$(SOURCES)))) \
	  $(filter-out %.cc,$(notdir $(patsubst %.c,%.o,$(SOURCES))))

LDFLAGS +=   -lEGL  -L. -L/usr/lib/x86_64-linux-gnu -lOpenGL

# Comment out to build without debug info.
CXXFLAGS += -g -I. 

SOURCES = tinyegl.cc

BIN = tinyegl

$(BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

.PHONY: clean
clean:
	rm -f $(BIN) *.o
