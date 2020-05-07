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
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;



/**
 * functions to get an initializer based on String name, 
 * an Initializer class, or lambda function
 * @author Jim Clarke
 */
public class Initializers {
    static Map<String, Supplier<Initializer> > map = new HashMap<String, Supplier<Initializer> >() 
        {{
           put("identity", Identity::new);
           put("ones",  Ones::new);
           put("zeros",  Zeros::new);
           put("glorot_normal",  GlorotNormal::new);
           put("glorot_uniform",  GlorotUniform::new);
           put("orthogonal",  Orthogonal::new);
           put("random_normal",  RandomNormal::new);
           put("random_uniform",  RandomUniform::new);
           put("truncated_normal",  TruncatedNormal::new);
           put("variance_scaling",  VarianceScaling::new);
           put("he_normal", HeNormal::new);
           put("he_uniform", HeUniform::new);
           put("lecun_normal", LeCunNormal::new);
           put("lecun_uniform", LeCunUniform::new);
        }};

     /**
      * Get an Initializer
      * @param initializerFunction either a String that identifies the Initializer, 
      * an Initializer class, or an Initializer object.
      * @return the Intializer object or null if not found.
      */
     public static Initializer get(Object initializerFunction) {
        return get(initializerFunction, null);
    }
    
     /**
      * Get an Initializer
      * @param si a lamda function
      * @return the Intializer object
      */
    public static Initializer get(Supplier<Initializer> si) {
         return si.get();
    }
    
    /**
     * Get an Initializer
     * @param initializerFunction
     * @param custom_functions a map of Initializer lambdas that will be queried 
     * if the initializer is not found in the standard keys
     * @return the Intializer object
     */
    public static Initializer get(Object initializerFunction, Map<String,  Supplier<Initializer>> custom_functions) {
        if(initializerFunction != null) {
            if(initializerFunction instanceof String) {
                String s = initializerFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Supplier<Initializer> function = map.get(s);
                if(function == null && custom_functions != null)
                    function = custom_functions.get(s);
                return function != null ? function.get() : null;
            }else if(initializerFunction instanceof Class ) {
                Class c = (Class)initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor();
                    return (Initializer)ctor.newInstance();
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
                "initializerFunction must be a symbolic name, Initializer, Supplier<Initializer> or a Class object");
    }
    
}
