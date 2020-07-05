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
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.keras.metrics.impl.Reduce;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class Mean extends Reduce {

    public Mean(Ops tf) {
        this(tf, null, null);
    }

    public Mean(Ops tf, DataType dType) {
        this(tf, null, dType);
    }

    public Mean(Ops tf, String name) {
        this(tf, name, null);
    }

    public Mean(Ops tf, String name, DataType dType) {
        super(tf, name, Reduction.WEIGHTED_MEAN, dType);
    }

}
