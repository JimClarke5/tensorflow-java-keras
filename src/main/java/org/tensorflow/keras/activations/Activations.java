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
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.keras.initializers.Initializers;

/**
 * Retrieve Activation functions based on symbolic name, lambda, or Class
 * @author Jim Clarke
 */
public class Activations {

    static Map<String, Supplier<Activation>> map = new HashMap<String, Supplier<Activation>>() {
        {
            put("relu", ReLU::new);
            put("elu", ELU::new);
            put("exponential", Exponential::new);
            put("hard_sigmoid", HardSigmoid::new);
            put("linear", Linear::new);
            put("selu", SELU::new);
            put("sigmoid", Sigmoid::new);
            put("softmax", Softmax::new);
            put("softplus", Softplus::new);
            put("softsign", Softsign::new);
            put("swish", Swish::new);
            put("tanh", Tanh::new);
        }
    };

    /**
     * Get an Activation
     *
     * @param initializerFunction either a String that identifies the
     * Activation, an Activation class, or an Activation object.
     * @return the activation object or null if not found.
     */
    public static Activation get(Object initializerFunction) {
        return get(initializerFunction, null);
    }

    /**
     * Get an Activation based on a lambda
     *
     * @param lamda a lamda function
     * @return the Intializer object
     */
    public static Activation get(Supplier<Activation> lamda) {
        return lamda.get();
    }

    /**
     * Get an Activation
     *
     * @param initializerFunction
     * @param custom_functions a map of Activation lambdas that will be queried
     * if the activation is not found in the standard keys
     * @return the Activation object
     */
    public static Activation get(Object initializerFunction, Map<String, Supplier<Activation>> custom_functions) {
        if (initializerFunction != null) {
            if (initializerFunction instanceof String) {
                String s = initializerFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Supplier<Activation> function = map.get(s);
                if (function == null && custom_functions != null) {
                    function = custom_functions.get(s);
                }
                return function != null ? function.get() : null;
            } else if (initializerFunction instanceof Class) {
                Class c = (Class) initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor();
                    return (Activation) ctor.newInstance();
                } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                    Logger.getLogger(Initializers.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else if (initializerFunction instanceof Activation) {
                return (Activation) initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
            }
        } else {
            return null;
        }

        throw new IllegalArgumentException(
                "initializerFunction must be a symbolic name, Activation, Supplier<Activation> or a Class object");
    }

    // Helper function calls to create the Activation
    
    /**
     * Applies the rectified linear unit activation function.
     * @return the ReLU activation
     */
    public static ReLU relu() {
        return new ReLU();
    }

    /**
     * Applies the rectified linear unit activation function.
     * @param alpha governs the slope for values lower than the threshold.
     * @param max_value sets the saturation threshold (the largest value the function will return).
     * @param threshold the threshold value of the activation function below which values will be damped or set to zero.
     * @return the ReLU activation
     */
    public static ReLU relu(double alpha, Double max_value, double threshold) {
        return new ReLU(alpha, max_value, threshold);
    }

    /**
     * Exponential linear unit.
     * @return the ELU activation
     */
    public static ELU elu() {
        return new ELU();
    }
    
    /**
     * Exponential linear unit.
     * @param alpha A scalar, slope of negative section.
     * @return the ELU activation
     */
    public static ELU elu(double alpha) {
        return new ELU(alpha);
    }

    /**
     *  Exponential activation function.
     * @return the exponential activation
     */
    public static Exponential exponential() {
        return new Exponential();
    }

    /**
     * Hard sigmoid activation function.
     * @return the Hard sigmoid activation function.
     */
    public static HardSigmoid hard_sigmoid() {
        return new HardSigmoid();
    }

    /**
     * Linear activation function.
     * @return the Linear activation function.
     */
    public static Linear linear() {
        return new Linear();
    }

    /**
     * Scaled Exponential Linear Unit (SELU).
     * @return the Scaled Exponential Linear Unit (SELU).
     */
    public static SELU selu() {
        return new SELU();
    }

    /**
     * Sigmoid activation function.
     * @return the Sigmoid activation function.
     */
    public static Sigmoid sigmoid() {
        return new Sigmoid();
    }

    /**
     * Softmax converts a real vector to a vector of categorical probabilities.
     * @return the Softmax activation function.
     */
    public static Softmax softmax() {
        return new Softmax();
    }
    
    /**
     * Softmax converts a real vector to a vector of categorical probabilities.
     * @param axis axis along which the softmax normalization is applied.
     * @return the Softmax activation function.
     */
    public static Softmax softmax(int axis) {
        return new Softmax(axis);
    }

    /**
     * Softplus activation function.
     * @return the Softplus activation function.
     */
    public static Softplus softplus() {
        return new Softplus();
    }

    /**
     * Softsign activation function.
     * @return the Softsign activation function.
     */
    public static Softsign softsign() {
        return new Softsign();
    }

    /**
     * Swish activation function.
     * @return the Swish activation function.
     */
    public static Swish swish() {
        return new Swish();
    }

    /**
     * Hyperbolic tangent activation function.
     * @return the Hyperbolic tangent activation function.
     */
    public static Tanh tanh() {
        return new Tanh();
    }

}
