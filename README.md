# tensorflow-java-keras

***!!! IMPORTANT NOTICE !!! This repository is UNDER CONSTRUCTION and does not yet host the code of the 
offical TensorFlow Java artifacts!***

***This project is dependent on the development build for Tensorlfow Java. 
Please refer to the [TensorFlow Java module](https://github.com/tensorflow/java) 
of the main repository for the actual code.***

## This project is meant to port the TensorFlow Keras module from Python to Java.



## This Repository

This repository is a Java Keras module that works with the latest TensorFlow java apis.

The working module name is tensorflow-java-keras.

The following  artifacts are required to build this module from
[TensorFlow Java module](https://github.com/tensorflow/java) 
:

* `tensorflow-core`
  * All artifacts that build up the core language bindings of TensorFlow for Java. 
  * Those artifacts provide the minimal support required to use the TensorFlow runtime on a JVM.
    
* `tensorflow-framework`
  * High-level APIs built on top of the core libraries to simplify neural network training and inference 
    using TensorFlow.
  
* `tensorflow-tools`
  * Utility libraries that do not depend on the TensorFlow runtime but are useful for machine learning purposes.
  
## Building Sources

To build all the artifacts, simply invoke the command `mvn install` at the root of this repository (or 
the Maven command of your choice). 


## Using Maven Artifacts

To include this module in your Maven application, you first need to add a dependencies on either the
`tensorflow-core` or `tensorflow-core-platform` artifacts. The former could be included multiple times
for different targeted systems by their classifiers, while the later includes them as dependencies for
`linux-x86_64`, `macosx-x86_64`, and `windows-x86_64`, with more to come in the future. There are also
`tensorflow-core-platform-mkl`, `tensorflow-core-platform-gpu`, and `tensorflow-core-platform-mkl-gpu`
artifacts that depend on artifacts with MKL and/or CUDA support enabled.

You should add the following dependencies:
```xml
        <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>tensorflow-core-api</artifactId>
            <version>0.1.0-SNAPSHOT</version>
          </dependency>
        <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>tensorflow-core-generator</artifactId>
            <version>0.1.0-SNAPSHOT</version>
        </dependency>
        <dependency>
            <groupId>org.tensorflow</groupId>
            <artifactId>tensorflow-core-platform</artifactId>
            <version>0.1.0-SNAPSHOT</version>
        </dependency>
        <dependency>
            <groupId>org.json</groupId>
            <artifactId>json</artifactId>
            <version>20190722</version>
        </dependency>
```



## TensorFlow Version Support

This table shows the mapping between different version of TensorFlow for Java and the core runtime libraries.

| TensorFlow Java Version  | TensorFlow Version |
| ------------- | ------------- |
| 0.1.0-SNAPSHOT  | 2.2.0  |



