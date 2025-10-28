include ../../src/Makefile.include

TARGET := primality
OBJS := Primality.o
DECLS := primality.decl.h
CI := primality.ci

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CHARMC) -language charm++ -o $@ $(OBJS)

$(OBJS): %.o: %.C $(DECLS)
	$(CHARMC) -c $<

$(DECLS): %.decl.h: %.ci
	$(CHARMC) $<

clean:
	-$(RM) $(TARGET) *.decl.h *.def.h *.o charmrun

.PHONY: all clean
