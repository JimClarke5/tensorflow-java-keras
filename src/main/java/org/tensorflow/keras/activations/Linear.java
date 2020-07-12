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
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

/**
 * Linear activation function.
 */
public class Linear<U extends TType> extends Activation<U> {

    /**
     * Create a linear activation.
     *
     * @param tf the TensorFlow Ops
     */
    public Linear(Ops tf) {
        super(tf);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<U> call(Operand<U> input) {
        return input;
    }

}
