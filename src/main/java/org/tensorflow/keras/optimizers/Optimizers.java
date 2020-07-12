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
package org.tensorflow.keras.optimizers;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.op.Ops;

/**
 * functions to get an initializer based on String name, an Initializer class,
 * or lambda function
 *
 * @author Jim Clarke
 */
public class Optimizers {

    static Map<String, Function<Ops, Optimizer>> map = new HashMap<String, Function<Ops, Optimizer>>() {
        {
            put("adadelta", tf -> new AdaDelta(tf));
            put("adagrad", tf -> new AdaGrad(tf));
            put("AdagradDA", tf -> new AdaGradDA(tf));
            put("adam", tf -> new Adam(tf));
            put("adamax", tf -> new Adamax(tf));
            put("ftrl", tf -> new Ftrl(tf));
            put("nadam", tf -> new Nadam(tf));
            put("rmsprop", tf -> new RMSProp(tf));
            put("sgd", tf -> new SGD(tf));
        }
    };

    /**
     * Get an Initializer
     *
     * @param initializerFunction either a String that identifies the
     * Initializer, an Initializer class, or an Initializer object.
     * @return the Intializer object or null if not found.
     */
    public static Optimizer get(Ops tf, Object initializerFunction) {
        return get(tf, initializerFunction, null);
    }

    /**
     * Get an Initializer
     *
     * @param si a lamda function
     * @return the Intializer object
     */
    public static Optimizer get(Ops tf, Function<Ops, Optimizer> func) {
        return func.apply(tf);
    }

    /**
     * Get an Initializer
     *
     * @param initializerFunction
     * @param custom_functions a map of Initializer lambdas that will be queried
     * if the initializer is not found in the standard keys
     * @return the Intializer object
     */
    public static Optimizer get(Ops tf, Object initializerFunction, Map<String, Function<Ops, Optimizer>> custom_functions) {
        if (initializerFunction != null) {
            if (initializerFunction instanceof String) {
                String s = initializerFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Function<Ops, Optimizer> function = map.get(s);
                if (function == null && custom_functions != null) {
                    function = custom_functions.get(s);
                }
                return function != null ? function.apply(tf) : null;
            } else if (initializerFunction instanceof Class) {
                Class c = (Class) initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor(Ops.class);
                    return (Optimizer) ctor.newInstance(tf);
                } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                    Logger.getLogger(Optimizers.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else if (initializerFunction instanceof Optimizer) {
                return (Optimizer) initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
            }
        } else {
            return null;
        }

        throw new IllegalArgumentException(
                "initializerFunction must be a symbolic name, Initializer, Supplier<Initializer> or a Class object");
    }

}
