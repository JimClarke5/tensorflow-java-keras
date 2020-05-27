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
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 *
 * @author Jim Clarke
 */
public class Zeros<U extends TType> extends Initializer<U> {
    
    /**
     * create an Initializer that sets all values to one.
     * @param tf the TensorFlow Ops
     */
    public Zeros(Ops tf) {
        super(tf);
    }
    
    /**
     * create an Initializer that sets all values to one.
     * 
     * @param tf the TensorFlow Ops
     * @param config a config object to initialize
     */
    public Zeros(Ops tf, Map<String, Object> config) {
        super(tf,config);
    }
    
    @Override
    public Operand<U> call(Operand<TInt64> dims, DataType<U> dtype) {
        return tf.zeros(dims, dtype);
        
    }
    
    
}
