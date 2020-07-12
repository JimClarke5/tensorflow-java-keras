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
 * Retrieve Activation functions based on symbolic name, lambda, Class, or
 * method
 */
public class Activations {

    /**
     * map of string names to instances of activations.
     */
    static Map<String, Function<Ops, Activation>> map = new HashMap<String, Function<Ops, Activation>>() {
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
     * @param tf the TensorFlow Ops
     * @param activationFunction either a String that identifies the Activation,
     * an Activation class, or an Activation object.
     * @return the activation instance or null if not found.
     */
    public static Activation get(Ops tf, Object activationFunction) {
        return get(tf, activationFunction, null);
    }

    /**
     * Get an Activation based on a lambda of the form: (Ops ops) -> create(Ops
     * ops)
     *
     * @param tf the TensorFlow Ops
     * @param lambda a lambda function
     * @return the Activation instance
     */
    public static Activation get(Ops tf, Function<Ops, Activation> lambda) {
        return lambda.apply(tf);
    }

    /**
     * Get an Activation based on a lambda of the form: () -> create()
     *
     * @param lambda a lambda function
     * @return the Activation instance
     */
    public static Activation get(Supplier<Activation> lambda) {
        return lambda.get();
    }

    /**
     * Get an Activation
     *
     * @param tf the TensorFlow Ops
     * @param activationFunction
     * @param custom_functions a map of Activation lambdas that will be queried
     * if the activation is not found in the standard keys
     * @return the Activation instance
     */
    public static Activation get(Ops tf, Object activationFunction, Map<String, Function<Ops, Activation>> custom_functions) {
        if (activationFunction != null) {
            if (activationFunction instanceof String) {
                String s = activationFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Function<Ops, Activation> function = map.get(s);
                if (function == null && custom_functions != null) {
                    function = custom_functions.get(s);
                }
                return function != null ? function.apply(tf) : null;
            } else if (activationFunction instanceof Class) {
                Class c = (Class) activationFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor(Ops.class);
                    return (Activation) ctor.newInstance(tf);
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

    // Helper function calls to create an Activation
    /**
     * Creates a rectified linear unit (ReLU) activation.
     *
     * @param tf the TensorFlow Ops
     * @return the ReLU activation
     */
    public static ReLU relu(Ops tf) {
        return new ReLU(tf);
    }

    /**
     * Creates a rectified linear unit (ReLU) activation.
     *
     * @param tf the TensorFlow Ops
     * @param alpha governs the slope for values lower than the threshold.
     * @param max_value sets the saturation threshold (the largest value the
     * function will return).
     * @param threshold the threshold value of the activation function below
     * which values will be damped or set to zero.
     * @return the ReLU activation
     */
    public static ReLU relu(Ops tf, double alpha, Double max_value, double threshold) {
        return new ReLU(tf, alpha, max_value, threshold);
    }

    /**
     * Creates an exponential linear unit(ELU) activation.
     *
     * @param tf the TensorFlow Ops
     * @return the ELU activation
     */
    public static ELU elu(Ops tf) {
        return new ELU(tf);
    }

    /**
     * Creates an exponential linear unit(ELU) activation.
     *
     * @param tf the TensorFlow Ops
     * @param alpha A scalar, slope of negative section.
     * @return the ELU activation
     */
    public static ELU elu(Ops tf, double alpha) {
        return new ELU(tf, alpha);
    }

    /**
     * Creates an exponential activation.
     *
     * @param tf the TensorFlow Ops
     * @return the exponential activation
     */
    public static Exponential exponential(Ops tf) {
        return new Exponential(tf);
    }

    /**
     * Creates a hard sigmoid activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Hard sigmoid activation instance.
     */
    public static HardSigmoid hard_sigmoid(Ops tf) {
        return new HardSigmoid(tf);
    }

    /**
     * Creates a linear activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Linear activation instance.
     */
    public static Linear linear(Ops tf) {
        return new Linear(tf);
    }

    /**
     * Creates a scaled exponential linear unit (SELU) activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Scaled Exponential Linear Unit (SELU).
     */
    public static SELU selu(Ops tf) {
        return new SELU(tf);
    }

    /**
     * Creates a sigmoid activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Sigmoid activation instance.
     */
    public static Sigmoid sigmoid(Ops tf) {
        return new Sigmoid(tf);
    }

    /**
     * Creates a softmax activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Softmax activation instance.
     */
    public static Softmax softmax(Ops tf) {
        return new Softmax(tf);
    }

    /**
     * Creates a softmax activation.
     *
     * @param tf the TensorFlow Ops
     * @param axis axis along which the softmax normalization is applied.
     * @return the Softmax activation instance.
     */
    public static Softmax softmax(Ops tf, int axis) {
        return new Softmax(tf, axis);
    }

    /**
     * Creates a softplus activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Softplus activation instance.
     */
    public static Softplus softplus(Ops tf) {
        return new Softplus(tf);
    }

    /**
     * Creates a softsign activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Softsign activation instance.
     */
    public static Softsign softsign(Ops tf) {
        return new Softsign(tf);
    }

    /**
     * Creates a swish activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Swish activation instance.
     */
    public static Swish swish(Ops tf) {
        return new Swish(tf);
    }

    /**
     * Creates a hyperbolic tangent (tanh) activation.
     *
     * @param tf the TensorFlow Ops
     * @return the Hyperbolic tangent activation instance.
     */
    public static Tanh tanh(Ops tf) {
        return new Tanh(tf);
    }

}
