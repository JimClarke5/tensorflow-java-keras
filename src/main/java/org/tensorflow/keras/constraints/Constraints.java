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
package org.tensorflow.keras.constraints;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * functions to get a constraint based on String name, an Constraint class, or
 * lambda function
 */
public class Constraints {

    static Map<String, Function<Ops, Constraint>> map = new HashMap<String, Function<Ops, Constraint>>() {
        {
            put("max_norm", tf -> new MaxNorm(tf));
            put("non_neg", tf -> new NonNeg(tf));
            put("unit_norm", tf -> new UnitNorm(tf));
            put("min_max_norm", tf -> new MinMaxNorm(tf));
            put("radial_constraint", tf -> new RadialConstraint(tf));
        }
    };

    /**
     * Get a Constraint
     *
     * @param tf the TensorFlow Ops
     * @param initializerFunction either a String that identifies the
     * Constraint, an Constraint class, or an Constraint object.
     * @return the Intializer object or null if not found.
     */
    public static Constraint get(Ops tf, Object initializerFunction) {
        return get(tf, initializerFunction, null);
    }

    /**
     * Get an Constraint using a lamda of the form (Ops ops) -> create(Ops ops)
     *
     * @param tf the TensorFlow Ops
     * @param lambda a lambda function
     * @return the Intializer
     */
    public static Constraint get(Ops tf, Function<Ops, Constraint> lambda) {
        return lambda.apply(tf);
    }

    /**
     * Get an Constraint using a lamda of the form () -> create()
     *
     * @param lambda a lamda function
     * @return the Intializer object
     */
    public static Constraint get(Supplier<Constraint> lambda) {
        return lambda.get();
    }

    /**
     * Get an Constraint
     *
     * @param tf the TensorFlow Ops
     * @param initializerFunction
     * @param custom_functions a map of Constraint lambdas that will be queried
     * if the initializer is not found in the standard keys
     * @return the Intializer object
     */
    public static Constraint get(Ops tf, Object initializerFunction, Map<String, Function<Ops, Constraint>> custom_functions) {
        if (initializerFunction != null) {
            if (initializerFunction instanceof String) {
                String s = initializerFunction.toString(); // do this for Java 8 rather than Pattern Matching for instanceof
                Function<Ops, Constraint> function = map.get(s);
                if (function == null && custom_functions != null) {
                    function = custom_functions.get(s);
                }
                return function != null ? function.apply(tf) : null;
            } else if (initializerFunction instanceof Class) {
                Class c = (Class) initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
                try {
                    Constructor ctor = c.getConstructor(Ops.class);
                    return (Constraint) ctor.newInstance(tf);
                } catch (NoSuchMethodException | InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException ex) {
                    Logger.getLogger(Constraints.class.getName()).log(Level.SEVERE, null, ex);
                }
            } else if (initializerFunction instanceof Constraint) {
                return (Constraint) initializerFunction; // do this for Java 8 rather than Pattern Matching for instanceof
            }
        } else {
            return null;
        }

        throw new IllegalArgumentException(
                "initializerFunction must be a symbolic name, Constraint, Function<Ops, Constraint > or a Class object");
    }

    /**
     * Apply a MaxNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> max_norm(Ops tf, Operand<T> weights) {
        MaxNorm instance = new MaxNorm(tf);
        return instance.call(weights);
    }

    /**
     * Apply a MaxNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param maxValue the maximum norm for the incoming weights.
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> max_norm(Ops tf, Operand<T> weights, float maxValue) {
        return max_norm(tf, weights, maxValue, MaxNorm.AXIS_DEFAULT);
    }

    /**
     * Apply a MaxNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param axis integer, axis along which to calculate weight norms.
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> max_norm(Ops tf, Operand<T> weights, int axis) {
        return max_norm(tf, weights, MaxNorm.MAX_VALUE_DEFAULT, axis);
    }

    /**
     * Apply a MaxNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param maxValue the maximum norm for the incoming weights.
     * @param axis integer, axis along which to calculate weight norms.
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> max_norm(Ops tf, Operand<T> weights, float maxValue, int axis) {
        MaxNorm instance = new MaxNorm(tf, maxValue, axis);
        return instance.call(weights);
    }

    /**
     * Apply a NonNeg constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> non_neg(Ops tf, Operand<T> weights) {
        NonNeg instance = new NonNeg(tf);
        return instance.call(weights);
    }

    /**
     * Apply a UnitNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> unit_norm(Ops tf, Operand<T> weights) {
        UnitNorm instance = new UnitNorm(tf);
        return instance.call(weights);
    }

    /**
     * Apply a UnitNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param axis axis along which to calculate weight norms.
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> unit_norm(Ops tf, Operand<T> weights, int axis) {
        UnitNorm instance = new UnitNorm(tf, axis);
        return instance.call(weights);
    }

    /**
     * Apply a MinMaxNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> min_max_norm(Ops tf, Operand<T> weights) {
        MinMaxNorm instance = new MinMaxNorm(tf);
        return instance.call(weights);
    }

    /**
     * Apply a MinMaxNorm constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param minValue the minimum norm for the incoming weights.
     * @param maxValue the maximum norm for the incoming weights.
     * @param rate the rate for enforcing the constraint.
     * @param axis integer, axis along which to calculate weight norms.
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> min_max_norm(Ops tf, Operand<T> weights,
            float minValue, float maxValue, float rate, int axis) {
        MinMaxNorm instance = new MinMaxNorm(tf);
        return instance.call(weights);
    }

    /**
     * Apply a RadialConstraint constraint against the weights
     *
     * @param tf the TensorFlow Ops
     * @param weights the weights
     * @param <T> the type of the weights
     * @return the constrained weights
     */
    public <T extends TNumber> Operand<T> radial_constraint(Ops tf, Operand<T> weights) {
        RadialConstraint instance = new RadialConstraint(tf);
        return instance.call(weights);
    }

}
