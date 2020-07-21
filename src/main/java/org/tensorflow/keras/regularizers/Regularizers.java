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
package org.tensorflow.keras.regularizers;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import static org.tensorflow.keras.regularizers.Regularizer.DEFAULT_REGULARIZATION_PENALTY;
import org.tensorflow.op.Ops;

/**
 *
 * @author jbclarke
 */
public class Regularizers {
    
    
    static Map<String, Function<Ops, Regularizer > > map = new HashMap<String, Function<Ops, Regularizer>>() 
        {{
            put("l1", tf -> new L1(tf));
            put("l2",  tf -> new L2(tf));
            put("l1_l2",  tf -> new l1_l2(tf));
        }};
    
    /**
      * Get a Regularizer
      * @param tf the TensorFlow Ops
      * @param regularizerFunction either a String that identifies the Regularizer, 
      * an Regularizer class, or an Regularizer object.
      * @return the Regularizer object or null if not found.
      */
     public static Regularizer get(Ops tf, Object regularizerFunction) {
        return get(tf, regularizerFunction, null);
    }
    
     /**
      * Get a Regularizer using a lamda of the form (Ops ops) -> create(Ops ops) 
      * @param tf the TensorFlow Ops
      * @param lambda a lambda function
      * @return the Regularizer
      */
    public static Regularizer get(Ops tf, Function<Ops, Regularizer > lambda) {
         return lambda.apply(tf);
    } 
    
     /**
      * Get a Regularizer using a lamda of the form () -> create() 
      * @param lambda a lamda function
      * @return the Regularizer object
      */
    public static Regularizer get( Supplier<Regularizer > lambda) {
         return lambda.get();
    }
    
    /**
     * Get a Regularizer
     * @param tf the TensorFlow Ops
     * @param regularizerFunction
     * @param custom_functions a map of Regularizer lambdas that will be queried 
     * if the regularizer is not found in the standard keys
     * @return the Regularizer object
     */
    public static Regularizer get(Ops tf, Object regularizerFunction, Map<String,  Function<Ops, Regularizer >> custom_functions) {
        if(regularizerFunction != null) {
            if(regularizerFunction instanceof String) {
                String s = regularizerFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Function<Ops, Regularizer > function = map.get(s);
                if(function == null && custom_functions != null)
                    function = custom_functions.get(s);
                return function != null ? function.apply(tf) : null;
            }else if(regularizerFunction instanceof Class ) {
                Class c = (Class)regularizerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor(Ops.class);
                    return (Regularizer)ctor.newInstance(tf);
                } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                    Logger.getLogger(Regularizers.class.getName()).log(Level.SEVERE, null, ex);
                }
            }else if(regularizerFunction instanceof Regularizer) {
                return (Regularizer)regularizerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
            }
        }else {
            return null;
        }
         
        throw new IllegalArgumentException(
                "regularizerFunction must be a symbolic name, Regularizer, Function<Ops, Regularizer > or a Class object");
    }
    
    public static L1L2 L1L2(Ops tf) {
        return new L1L2(tf);
    }
    
    public static L1L2 L1L2(Ops tf, Float l1, Float l2) {
        return new L1L2(tf, l1, l2);
    }
    
    public static l1_l2 l1_l2(Ops tf) {
        return new l1_l2(tf);
    }
    
    public static l1_l2 l1_l2(Ops tf, float l1, float l2) {
        return new l1_l2(tf, l1, l2);
    }
    
    public static L1 l1(Ops tf) {
        return new L1(tf);
    }
    
    public static L1 l1(Ops tf, float l1) {
        return new L1(tf, l1);
    }
    
    public static L2 l2(Ops tf) {
        return new L2(tf);
    }
    
    public static L2 l2(Ops tf, float l2) {
        return new L2(tf,l2);
    }
    
}
