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
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ReduceSum;
import org.tensorflow.types.family.TNumber;

/**
 *
 * @author jbclarke
 */
public class MinMaxNorm extends Constraint {

    public static final float MIN_VALUE_DEFAULT = 0.0F;
    public static final float MAX_VALUE_DEFAULT = 1.0F;
    public static final float RATE_DEFAULT = 1.0F;
    public static final int AXIS_DEFAULT = 0;

    /**
     * the minimum norm for the incoming weights. Default is 0.0.
     */
    private final float minValue;
    /**
     * the maximum norm for the incoming weights. Default is 1.0.
     */
    private final float maxValue;

    /**
     * rate for enforcing the constraint: weights will be rescaled to yield (1 -
     * rate) * norm + rate * norm.clip(min_value, max_value). Effectively, this
     * means that rate=1.0 stands for strict enforcement of the constraint,
     * while rate<1.0 means that weights will be rescaled at each step to slowly
     * move towards a value inside the desired interval. Default is 1.0;
     */
    private final float rate;

    /**
     * integer, axis along which to calculate weight norms. Default is 0.
     */
    private final int axis;

    /**
     * Create a MaxNorm constraint
     *
     * @param tf the TensorFlow Ops
     */
    public MinMaxNorm(Ops tf) {
        this(tf, MIN_VALUE_DEFAULT, MAX_VALUE_DEFAULT, RATE_DEFAULT, AXIS_DEFAULT);
    }

    /**
     * Create a MaxNorm constraint
     *
     * @param tf the TensorFlow Ops
     * @param minValue the minimum norm for the incoming weights.
     * @param maxValue the maximum norm for the incoming weights.
     */
    public MinMaxNorm(Ops tf, float minValue, float maxValue) {
        this(tf, minValue, maxValue, RATE_DEFAULT, AXIS_DEFAULT);
    }

    /**
     * Create a MaxNorm constraint
     *
     * @param tf the TensorFlow Ops
     * @param minValue the minimum norm for the incoming weights.
     * @param maxValue the maximum norm for the incoming weights.
     * @param rate the rate for enforcing the constraint.
     * @param axis integer, axis along which to calculate weight norms.
     */
    public MinMaxNorm(Ops tf, float minValue, float maxValue, float rate, int axis) {
        super(tf);
        this.minValue = minValue;
        this.maxValue = maxValue;
        this.rate = rate;
        this.axis = axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> weights) {
        DataType dType = weights.asOutput().dataType();
        Operand<T> norms = K.sqrt(tf,
                tf.reduceSum(tf.math.square(weights), tf.constant(getAxis()), ReduceSum.keepDims(Boolean.TRUE))
        );
        Operand desired = tf.math.add(tf.math.mul(tf.dtypes.cast(tf.constant(this.getRate()), dType),
                        K.clip(tf, norms, this.getMinValue(), this.getMaxValue())
                ),
                tf.math.mul(tf.math.sub(tf.dtypes.cast(tf.constant(1), dType),
                                tf.dtypes.cast(tf.constant(this.getRate()), dType)),
                        norms
                )
        );

        return tf.math.mul(weights,
                tf.math.div(desired, tf.math.add(
                        K.epsilonConstant(tf, dType), norms)));
    }

    /**
     * @return the minValue
     */
    public float getMinValue() {
        return minValue;
    }

    /**
     * @return the maxValue
     */
    public float getMaxValue() {
        return maxValue;
    }

    /**
     * @return the rate
     */
    public float getRate() {
        return rate;
    }

    /**
     * @return the axis
     */
    public int getAxis() {
        return axis;
    }

}
