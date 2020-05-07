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

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.ShapeUtils;
import org.tensorflow.keras.utils.TypeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.ReduceSum;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.family.TType;

/**
 * Softmax converts a real vector to a vector of categorical probabilities.
 * @author Jim Clarke
 */
public class Softmax <U extends TType> extends Activation<U> {
     private static final int AXIS_DEFAULT = -1;
     
     private final int axis;
     
     /**
      * Create a softmax activation where  the default axis is -1
      * which indicates the last dimension.
      */
     public Softmax() {
         this.axis = AXIS_DEFAULT;
     }
     
     /**
      *  Create a Softmax activation
      * @param axis  The dimension softmax would be performed on. 
      */
     public Softmax(int axis) {
         this.axis = axis;
     }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<U> call(Ops tf, Operand<U> input) {
        assert TypeUtils.isFloating(input.asTensor().dataType()): 
                "Must be a Floating Point DataType: " + input.asTensor().dataType();
        Shape shape = ShapeUtils.getShape(input.asTensor());
        
        if(shape.numDimensions() == 2) {
            return tf.nn.softmax((Operand)input);
        }else {
            Operand e = tf.math.exp(input);
             ReduceSum.Options option = ReduceSum.keepDims(Boolean.TRUE);
            Operand s = tf.reduceSum(input, tf.constant(0), option);
            return tf.math.div(e, s);
        }
    }
    
}
