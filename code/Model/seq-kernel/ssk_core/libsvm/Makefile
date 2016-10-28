.SUFFIXES: .class .java
FILES =	svm.class svm_model.class svm_node.class \
	svm_parameter.class svm_problem.class \
	CustomKernel.class

JAVAC = javac
JAVAC_FLAGS = -classpath ..

all: $(FILES)
	jar cvf libsvm.jar *.class

.java.class:
	$(JAVAC) $(JAVAC_FLAGS) $<

clean:
	rm -f *.class *.jar *~ *~

dist: clean all
	rm *.class
