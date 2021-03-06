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

import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

/**
 * Swish activation function.
 *
 * @param <T> the data type of the activation
 */
public class Swish<T extends TType> extends Activation<T> {

    /**
     * Create a Swish activation
     *
     * @param tf the TensorFlow Ops
     */
    public Swish(Ops tf) {
        super(tf);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<T> call(Operand<T> input) {
        assert TypeUtils.isFloating(input.asTensor().dataType()) :
                "Must be a Floating Point DataType: " + input.asTensor().dataType();
        // TODO What about the "grad" return from python tensorflow impl?
        return tf.math.mul(input, tf.math.sigmoid(input));
    }

}
