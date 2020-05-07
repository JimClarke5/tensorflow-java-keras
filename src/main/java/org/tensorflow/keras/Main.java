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
package org.tensorflow.keras;

import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.random.RandomUniform;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class Main {

    public static void main(String[] args) {
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            long[] seeds = new long[2];
            seeds[0] = 1000L;
            
            Operand<TFloat32> ops = tf.random.statelessRandomUniform(
                    tf.constant(shape.asArray()), tf.constant(seeds), TFloat32.DTYPE);
            
            ops.asTensor().data().scalars().forEach(s -> System.out.print(s.getFloat() + ", "));
            System.out.println();
            
        }
    }
}
