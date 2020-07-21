/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.initializers;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.op.Ops;



/**
 * functions to get an initializer based on String name, 
 * an Initializer class, or lambda function
 */
public class Initializers {
    static Map<String, Function<Ops, Initializer > > map = new HashMap<String, Function<Ops, Initializer>>() 
        {{
           put("identity", tf -> new Identity(tf));
           put("ones",  tf -> new Ones(tf));
           put("zeros",  tf -> new Zeros(tf));
           put("glorot_normal",  tf -> new GlorotNormal(tf));
           put("glorot_uniform",  tf -> new GlorotUniform(tf));
           put("orthogonal",  tf -> new Orthogonal(tf));
           put("random_normal",  tf -> new RandomNormal(tf));
           put("random_uniform",  tf -> new RandomUniform(tf));
           put("truncated_normal",  tf -> new TruncatedNormal(tf));
           put("variance_scaling",  tf -> new VarianceScaling(tf));
           put("he_normal", tf -> new HeNormal(tf));
           put("he_uniform", tf -> new HeUniform(tf));
           put("lecun_normal", tf -> new LeCunNormal(tf));
           put("lecun_uniform", tf -> new LeCunUniform(tf));
        }};

     /**
      * Get an Initializer
      * @param tf the TensorFlow Ops
      * @param initializerFunction either a String that identifies the Initializer, 
      * an Initializer class, or an Initializer object.
      * @return the Intializer object or null if not found.
      */
     public static Initializer get(Ops tf, Object initializerFunction) {
        return get(tf, initializerFunction, null);
    }
    
     /**
      * Get an Initializer using a lamda of the form (Ops ops) -> create(Ops ops) 
      * @param tf the TensorFlow Ops
      * @param lambda a lambda function
      * @return the Intializer
      */
    public static Initializer get(Ops tf, Function<Ops, Initializer > lambda) {
         return lambda.apply(tf);
    } 
    
     /**
      * Get an Initializer using a lamda of the form () -> create() 
      * @param tf the TensorFlow Ops
      * @param lambda a lamda function
      * @return the Intializer object
      */
    public static Initializer get( Supplier<Initializer > lambda) {
         return lambda.get();
    }
    
    /**
     * Get an Initializer
     * @param tf the TensorFlow Ops
     * @param initializerFunction
     * @param custom_functions a map of Initializer lambdas that will be queried 
     * if the initializer is not found in the standard keys
     * @return the Intializer object
     */
    public static Initializer get(Ops tf, Object initializerFunction, Map<String,  Function<Ops, Initializer >> custom_functions) {
        if(initializerFunction != null) {
            if(initializerFunction instanceof String) {
                String s = initializerFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Function<Ops, Initializer > function = map.get(s);
                if(function == null && custom_functions != null)
                    function = custom_functions.get(s);
                return function != null ? function.apply(tf) : null;
            }else if(initializerFunction instanceof Class ) {
                Class c = (Class)initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor(Ops.class);
                    return (Initializer)ctor.newInstance(tf);
                } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                    Logger.getLogger(Initializers.class.getName()).log(Level.SEVERE, null, ex);
                }
            }else if(initializerFunction instanceof Initializer) {
                return (Initializer)initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
            }
        }else {
            return null;
        }
         
        throw new IllegalArgumentException(
                "initializerFunction must be a symbolic name, Initializer, Function<Ops, Initializer > or a Class object");
    }
    
}
