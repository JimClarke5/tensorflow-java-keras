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
package org.tensorflow.keras.utils;

import java.util.ArrayList;
import java.util.List;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.NdArray;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.family.TType;

/**
 *
 * @author Jim Clarke
 */
public class ShapeUtils {
    
    /**
     * Convert shape operand back to a Shape object
     * @param dims the Operand containing the shape values
     * @return 
     */
    public static Shape getShape(Operand<TInt64> dims) {
        // TODO is there a better way?
        final List<Long> dimList = new ArrayList<>();
        dims.data().scalars().forEach(s -> dimList.add(s.getLong()));
        long[] longDims = new long[dimList.size()];
        for(int i = 0; i < dimList.size(); i++) {
            longDims[i] = dimList.get(i);
        }
        return  Shape.of(longDims);
    }
    
    /** 
     * get the shape for the data within a Tensor
     * @param tensor the tensor
     * @return the Shape of the tensor's data;
     */
    public static Shape getShape(Tensor tensor) {
        NdArray data = (NdArray) tensor.data();
        return  data.shape();
        
    }
}
