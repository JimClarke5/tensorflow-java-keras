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
package org.tensorflow.keras.constraints;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.utils.SmartCond;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Stack;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.family.TType;

/**
 * Constrains Conv2D kernel weights to be the same for each radius.
 */
public class RadialConstraint extends Constraint {

    /**
     * Create a RadialConstraint
     *
     * @param tf the TensorFlow Ops
     */
    public RadialConstraint(Ops tf) {
        super(tf);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TType> Operand<T> call(Operand<T> weights) {
        DataType dType = weights.asOutput().dataType();
        Shape wShape = weights.asOutput().shape();
        if (wShape.numDimensions() != 4) {
            throw new IllegalArgumentException(
                    String.format("The weight tensor must be of rank 4, but is of shape: %s", wShape.toString()));
        }

        long width = wShape.size(0);
        long height = wShape.size(1);
        long channels = wShape.size(2);
        long kernels = wShape.size(3);

        Shape newShape = Shape.of(width, height, channels * kernels);
        weights = tf.reshape(weights, tf.constant(newShape));
        weights = K.map(
                tf.stack(tf.unstack(weights, -1L), Stack.axis(0L)),
                kernel -> kernelConstraint(kernel));
        weights = tf.reshape(tf.stack(tf.unstack(weights, 0L), Stack.axis(-1L)), tf.constant(wShape));
        return weights;
    }

    private Operand kernelConstraint(Operand kernel) {
        DataType dType = kernel.asOutput().dataType();
        Operand<TInt32> padding = tf.constant(new int[][]{{1, 1}, {1, 1}});
        Operand<TInt32> kernelShape = tf.shape.size(kernel, tf.constant(0));
        Operand<TInt32> start = tf.math.div(kernelShape, tf.constant(2));
        // TODO 
        // kernel[start-1, start-1]
        Operand startMinus1 = tf.math.sub(start, tf.constant(1));
        Operand kernelNew = SmartCond.select(tf, 
                tf.dtypes.cast(tf.math.floorMod(kernelShape, tf.constant(2)), TBool.DTYPE), 
                () -> tf.slice(tf.slice(kernel, startMinus1, tf.constant(1)), startMinus1, tf.constant(1)), 
                () -> tf.math.add(tf.slice(tf.slice(kernel, startMinus1, tf.constant(1)), startMinus1, tf.constant(1)), 
                        tf.zeros(tf.constant(Shape.of(2,2)), dType)));
        
        Operand index =   SmartCond.select(tf, 
                tf.dtypes.cast(tf.math.floorMod(kernelShape, tf.constant(2)), TBool.DTYPE),       
                () -> tf.constant(0), 
                () -> tf.constant(1));
        
        //TODO
        /***********
        _, kernel_new = control_flow_ops.while_loop(
        while_condition,
        body_fn,
        [index, kernel_new],
        shape_invariants=[index.get_shape(),
                          tensor_shape.TensorShape([None, None])])
        *******************/
        return kernel;
    }
    
    private Operand body_fn(Operand start, int i, Operand kernel, Operand<TInt32> padding) {
        Operand<TInt32> startPlusi = tf.math.add(start, tf.constant(i));
        return tf.pad(kernel, padding,
                tf.slice(tf.slice(kernel, startPlusi, tf.constant(1)), startPlusi, tf.constant(1)));
    }

}