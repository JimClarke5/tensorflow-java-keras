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
 * Constrains the weights incident to each hidden unit to have unit norm.
 */
public class UnitNorm extends Constraint {

    public static final int AXIS_DEFAULT = 0;

    /**
     * integer, axis along which to calculate weight norms. Default is 0.
     */
    private final int axis;

    /**
     * Create a UnitNorm Constraint
     *
     * @param tf the TensorFlow Ops
     */
    public UnitNorm(Ops tf) {
        this(tf, AXIS_DEFAULT);
    }

    /**
     * Create a UnitNorm Constraint
     *
     * @param tf the TensorFlow Ops
     * @param axis axis along which to calculate weight norms.
     */
    public UnitNorm(Ops tf, int axis) {
        super(tf);
        this.axis = axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> weights) {
        DataType dType = weights.asOutput().dataType();

        return tf.math.div(weights,
                tf.math.add(tf.dtypes.cast(tf.constant(K.epsilon()), dType),
                        K.sqrt(tf,
                                tf.reduceSum(tf.math.square(weights), tf.constant(getAxis()), ReduceSum.keepDims(Boolean.TRUE))
                        )
                )
        );
    }

    /**
     * @return the axis
     */
    public int getAxis() {
        return axis;
    }

}
