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
package org.tensorflow.keras.regularizers;

import org.tensorflow.Operand;
import org.tensorflow.keras.backend.K;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;
import org.tensorflow.types.family.TType;

/**
 *
 * @author jbclarke
 */
public class L1L2 extends Regularizer {
    private final Float l1;
    private final Float l2;
    
    public L1L2(Ops tf) {
       this(tf, 0.f, 0.f);
    }
    
    public L1L2(Ops tf, Float l1, Float l2) {
        super(tf);
        this.l1 = l1;
        this.l2 = l2;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TType> Operand<T> call(Operand<T> input) {
        if(this.getL1() == null && this.getL2() == null) {
            return tf.dtypes.cast(tf.constant(0), input.asOutput().dataType());
        }
        Operand<T> regularization = tf.dtypes.cast(tf.constant(0), input.asOutput().dataType());
        
        
        if(this.getL1() != null && this.getL1() != 0.f) {
            Operand<T>  l1Op = tf.dtypes.cast(tf.constant(this.getL1()), input.asOutput().dataType());
            regularization = tf.math.add(regularization, 
                tf.math.mul(l1Op, tf.reduceSum(tf.math.abs((Operand<TNumber>)input), K.allAxis(tf, input)))
            );
        }
        
        if(this.getL2() != null && this.getL2() != 0.f) {
            Operand<T>  l2Op = tf.dtypes.cast(tf.constant(this.getL2()), input.asOutput().dataType());
            regularization = tf.math.add(regularization, 
                tf.math.mul(l2Op, tf.reduceSum(tf.math.square((Operand<TNumber>)input), K.allAxis(tf, input)))
            );
        }
        
        return regularization;
    }

    /**
     * @return the l1
     */
    public Float getL1() {
        return l1;
    }

    /**
     * @return the l2
     */
    public Float getL2() {
        return l2;
    }
    
}
