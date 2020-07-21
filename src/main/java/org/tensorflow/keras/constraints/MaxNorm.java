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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.metrics.impl.MetricsImpl;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ReduceSum;
import org.tensorflow.types.family.TNumber;

/**
 * MinMaxNorm weight constraint.
 */
public class MaxNorm extends Constraint {

    public static final float MAX_VALUE_DEFAULT = 2.0f;
    public static final int AXIS_DEFAULT = 0;

    /**
     * the maximum norm for the incoming weights. Default is 2.
     */
    private final float maxValue;
    /**
     * integer, axis along which to calculate weight norms. Default is 0.
     */
    private final int axis;

    /**
     * Create a MaxNorm constraint
     *
     * @param tf the TensorFlow Ops
     */
    public MaxNorm(Ops tf) {
        this(tf, MAX_VALUE_DEFAULT, AXIS_DEFAULT);
    }
    
    /**
     * Create a MaxNorm constraint
     *
     * @param tf the TensorFlow Ops
     * @param maxValue the maximum norm for the incoming weights.
     */
    public MaxNorm(Ops tf, float maxValue) {
        this(tf, maxValue, AXIS_DEFAULT);
    }

    /**
     * Create a MaxNorm constraint
     *
     * @param tf the TensorFlow Ops
     * @param maxValue the maximum norm for the incoming weights.
     * @param axis integer, axis along which to calculate weight norms.
     */
    public MaxNorm(Ops tf, float maxValue, int axis) {
        super(tf);
        this.maxValue = maxValue;
        this.axis = axis;
    }
    
    

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> weights) {
        DataType dType = weights.asOutput().dataType();
        Operand<T> norms = K.sqrt(tf,tf.reduceSum(tf.math.square(weights), tf.constant(getAxis()), ReduceSum.keepDims(Boolean.TRUE)) );
        Operand desired = K.clip(tf, norms, 0f, this.getMaxValue());
        
        return  tf.math.mul(weights, tf.math.div(desired, tf.math.add(
                        K.epsilonConstant(tf, dType), norms)) );
    }

    /**
     * @return the maxValue
     */
    public float getMaxValue() {
        return maxValue;
    }

    /**
     * @return the axis
     */
    public int getAxis() {
        return axis;
    }

}
