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
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

/**
 * Exponential linear unit.
 * @author Jim Clarke
 */
public class ELU <U extends TType> extends Activation<U> {
     private static final double ALPHA_DEFAULT = 1.0;
     
     /**
      * A scalar, slope of negative section.
      */
     private final double alpha;
     
     /** 
      * creates a new ELU with alpha=1.0
      */
     public ELU() {
         this.alpha = ALPHA_DEFAULT;
     }
     
     /**
      * creates a new ELU 
      * @param alpha A scalar, slope of negative section.
      */
     public ELU(double alpha) {
         this.alpha = alpha;
     }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand<U> call(Ops tf, Operand<U> input) {
        Operand result = tf.nn.elu((Operand)input);
        if(alpha == 1.0)
          return result;
        else {
            DataType dtype = input.asTensor().dataType();
            // return array_ops.where_v2(x > 0, res, alpha * res)
            Operand y = tf.math.mul(result, tf.dtypes.cast(tf.constant(alpha), dtype));
            Operand cond = tf.math.greater((Operand)result, tf.dtypes.cast(tf.constant(0), dtype));
            return tf.select(cond, result, y);
          
        }
    }
    
}
