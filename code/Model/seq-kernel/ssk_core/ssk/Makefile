.SUFFIXES: .class .java
FILES =	SubsequenceKernel.class FeatureDictionary.class \
	InstanceExample.class intex_node.class

JAVAC = javac
JAVAC_FLAGS = -classpath ..:../libsvm/libsvm.jar

all: $(FILES)
	jar cvf ssk.jar *.class

.java.class:
	$(JAVAC) $(JAVAC_FLAGS) $<

clean:
	rm -f *.class *.jar *~ *~

dist: clean all
	rm *.class
