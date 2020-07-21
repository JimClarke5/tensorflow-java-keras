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
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 * Constrains the weights to be non-negative.
 */
public class NonNeg extends Constraint {

    /**
     * Create a NonNeg constraint
     *
     * @param tf the TensorFlow Ops
     */
    public NonNeg(Ops tf) {
        super(tf);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TType> Operand<T> call(Operand<T> weights) {
        DataType dType = weights.asOutput().dataType();
        return tf.math.mul(weights,
                tf.dtypes.cast(tf.math.greaterEqual((Operand<TNumber>)weights, tf.dtypes.cast(tf.constant(0), dType)), dType)
        );
    }

}
