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
package org.tensorflow.keras.activations;

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
 * Retrieve Activation functions based on symbolic name, lambda, or Class
 * @author Jim Clarke
 */
public class Activations {

    static Map<String, Function<Ops, Activation >> map = new HashMap<String,Function<Ops, Activation >>() {
        {
            put("relu", tf -> new ReLU(tf));
            put("elu", tf -> new ELU(tf));
            put("exponential", tf -> new Exponential(tf));
            put("hard_sigmoid", tf -> new HardSigmoid(tf));
            put("linear", tf -> new Linear(tf));
            put("selu", tf -> new SELU(tf));
            put("sigmoid", tf -> new Sigmoid(tf));
            put("softmax", tf -> new Softmax(tf));
            put("softplus", tf -> new Softplus(tf));
            put("softsign", tf -> new Softsign(tf));
            put("swish", tf -> new Swish(tf));
            put("tanh", tf -> new Tanh(tf));
        }
    };

    /**
     * Get an Activation
     *
     * @param activationFunction either a String that identifies the
     * Activation, an Activation class, or an Activation object.
     * @return the activation object or null if not found.
     */
    public static Activation get(Ops tf, Object activationFunction) {
        return get(tf, activationFunction, null);
    }

    /**
     * Get an Activation based on a lambda
     *
     * @param tf
     * @param lambda a lambda function
     * @return the Intializer object
     */
    public static Activation get(Ops tf, Function<Ops, Activation > lambda) {
        return lambda.apply(tf);
    }
    
      /**
      * Get an Activation
      * @param lambda a lambda function
      * @return the Intializer object
      */
    public static Activation get( Supplier<Activation > lambda) {
         return lambda.get();
    }

    /**
     * Get an Activation
     *
     * @param activationFunction
     * @param custom_functions a map of Activation lambdas that will be queried
     * if the activation is not found in the standard keys
     * @return the Activation object
     */
    public static Activation get(Ops tf, Object activationFunction, Map<String, Function<Ops, Activation > > custom_functions) {
        if (activationFunction != null) {
            if (activationFunction instanceof String) {
                String s = activationFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Function<Ops, Activation > function = map.get(s);
                if (function == null && custom_functions != null) {
                    function = custom_functions.get(s);
                }
                return function != null ? function.apply(tf) : null;
            } else if (activationFunction instanceof Class) {
                Class c = (Class) activationFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor(Ops.class);
                    return (Activation)ctor.newInstance(tf);
                } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                    Logger.getLogger(Activations.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else if (activationFunction instanceof Activation) {
                return (Activation) activationFunction; // do this for Java 8 rather than Pattern Matching for instanceof
            }
        } else {
            return null;
        }

        throw new IllegalArgumentException(
                "activationFunction must be a symbolic name, Activation, Supplier<Activation> or a Class object");
    }

    // Helper function calls to create the Activation
    
    /**
     * Applies the rectified linear unit activation function.
     * @return the ReLU activation
     */
    public static ReLU relu(Ops tf) {
        return new ReLU(tf);
    }

    /**
     * Applies the rectified linear unit activation function.
     * @param alpha governs the slope for values lower than the threshold.
     * @param max_value sets the saturation threshold (the largest value the function will return).
     * @param threshold the threshold value of the activation function below which values will be damped or set to zero.
     * @return the ReLU activation
     */
    public static ReLU relu(Ops tf, double alpha, Double max_value, double threshold) {
        return new ReLU(tf, alpha, max_value, threshold);
    }

    /**
     * Exponential linear unit.
     * @return the ELU activation
     */
    public static ELU elu(Ops tf) {
        return new ELU(tf);
    }
    
    /**
     * Exponential linear unit.
     * @param alpha A scalar, slope of negative section.
     * @return the ELU activation
     */
    public static ELU elu(Ops tf, double alpha) {
        return new ELU(tf, alpha);
    }

    /**
     *  Exponential activation function.
     * @return the exponential activation
     */
    public static Exponential exponential(Ops tf) {
        return new Exponential(tf);
    }

    /**
     * Hard sigmoid activation function.
     * @return the Hard sigmoid activation function.
     */
    public static HardSigmoid hard_sigmoid(Ops tf) {
        return new HardSigmoid(tf);
    }

    /**
     * Linear activation function.
     * @return the Linear activation function.
     */
    public static Linear linear(Ops tf) {
        return new Linear(tf);
    }

    /**
     * Scaled Exponential Linear Unit (SELU).
     * @return the Scaled Exponential Linear Unit (SELU).
     */
    public static SELU selu(Ops tf) {
        return new SELU(tf);
    }

    /**
     * Sigmoid activation function.
     * @return the Sigmoid activation function.
     */
    public static Sigmoid sigmoid(Ops tf) {
        return new Sigmoid(tf);
    }

    /**
     * Softmax converts a real vector to a vector of categorical probabilities.
     * @return the Softmax activation function.
     */
    public static Softmax softmax(Ops tf) {
        return new Softmax(tf);
    }
    
    /**
     * Softmax converts a real vector to a vector of categorical probabilities.
     * @param axis axis along which the softmax normalization is applied.
     * @return the Softmax activation function.
     */
    public static Softmax softmax(Ops tf, int axis) {
        return new Softmax(tf, axis);
    }

    /**
     * Softplus activation function.
     * @return the Softplus activation function.
     */
    public static Softplus softplus(Ops tf) {
        return new Softplus(tf);
    }

    /**
     * Softsign activation function.
     * @return the Softsign activation function.
     */
    public static Softsign softsign(Ops tf) {
        return new Softsign(tf);
    }

    /**
     * Swish activation function.
     * @return the Swish activation function.
     */
    public static Swish swish(Ops tf) {
        return new Swish(tf);
    }

    /**
     * Hyperbolic tangent activation function.
     * @return the Hyperbolic tangent activation function.
     */
    public static Tanh tanh(Ops tf) {
        return new Tanh(tf);
    }

}
