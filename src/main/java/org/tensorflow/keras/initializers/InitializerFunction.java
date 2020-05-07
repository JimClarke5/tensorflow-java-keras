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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 * A Functional Interface for Initializers
 * 
 * @author Jim Clarke
 * @param <U> The Type for the call operation
 */

@FunctionalInterface
public interface InitializerFunction<U extends TType> {
    
    
    /**
     *  The call operation for the initializer
     * 
     * @param tf the tensorflow Ops
     * @param dims the shape dimensions
     * @param dtype the data type
     * @return  An operand for the initialization.
     */
    public Operand<U> call(Ops tf, Operand<TInt64> dims, DataType<U> dtype);
    
}
