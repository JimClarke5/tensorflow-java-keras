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
package org.tensorflow.keras.initializers;

import java.util.Map;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 * Initializer that generates tensors initialized to 1.
 * @author Jim Clarke
 */
public class Ones<U extends TType> extends Initializer<U> {
    
    /**
     * create an Initializer that sets all values to one.
     * 
     * @param tf the TensorFlow Ops
     */
    public Ones(Ops tf) {
        super(tf);
    }
    
    /**
     * create an Initializer that sets all values to one.
     * 
     * @param tf the TensorFlow Ops
     * @param config a config object to initialize
     */
    public Ones(Ops tf, Map<String, Object> config) {
        super(tf, config);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<U> call(Operand<TInt64> dims, DataType<U> dtype) {
        assert(TypeUtils.isNumeric(dtype) || TypeUtils.isBoolean(dtype)) : 
                "DataType must be numeric or boolean: " + dtype.name();
        return tf.fill(dims, tf.dtypes.cast(tf.constant(1.0), dtype));
    }


    
}
