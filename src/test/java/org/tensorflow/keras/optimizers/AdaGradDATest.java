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
package org.tensorflow.keras.optimizers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.optimizers.Optimizer;
import static org.tensorflow.keras.optimizers.AdaGradDA.INITIAL_ACCUM_KEY;
import static org.tensorflow.keras.optimizers.AdaGradDA.LEARNING_RATE_KEY;
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class AdaGradDATest {

    int index;

    public AdaGradDATest() {
    }

    @BeforeAll
    public static void setUpClass() {
    }

    @AfterAll
    public static void tearDownClass() {
    }

    @BeforeEach
    public void setUp() {
    }

    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of create method, of class AdaGradDA.
     */
    @Test
    public void testCreate() {
        try (Graph graph = new Graph()) {
            Map<String, Object> config = new HashMap<>();
            config.put(NAME_KEY, "AdaDelta");
            config.put(LEARNING_RATE_KEY, 2.0F);
            config.put(INITIAL_ACCUM_KEY, 0.1F);
            AdaGradDA expResult = new AdaGradDA(graph, 2.0F, 0.1F, 0.0F, 0.0F);
            AdaGradDA result = AdaGradDA.create(graph, config);
            assertEquals(expResult.getConfig(), result.getConfig());
        }
    }

    @Test
    public void testBasic() {
        float[] var0_init = {0.0F, 0.0F};
        float[] var1_init = {0.0F, 0.0F};
        float[] grads0_init = {0.1F, 0.2F};
        float[] grads1_init = {0.01F, 0.02F};
        float epsilon = 1e-8F;
        float epsilon1 = 1e-5F;
        try (Graph graph = new Graph(); Session sess = new Session(graph)) {
            Ops tf = Ops.create(graph).withName("test");

            Shape shape0 = Shape.of(var0_init.length);
            Shape shape1 = Shape.of(var1_init.length);
            Variable<TFloat32> var0 = tf.withName("var0").variable(shape0, TFloat32.DTYPE);
            Variable<TFloat32> var1 = tf.withName("var1").variable(shape1, TFloat32.DTYPE);

            Assign<TFloat32> var0Initializer = tf.assign(var0, tf.constant(var0_init));
            Assign<TFloat32> var1Initializer = tf.assign(var1, tf.constant(var1_init));

            Constant<TFloat32> grads0 = tf.constant(grads0_init);
            Constant<TFloat32> grads1 = tf.constant(grads1_init);

            /* initialize the local variables */
            sess.runner().addTarget(var0Initializer).run();
            sess.runner().addTarget(var1Initializer).run();

            float learningRate = 3.0F;

            AdaGrad instance = new AdaGrad(graph, learningRate);

            /* build the GradsAnvVars */
            List gradsAndVars = new ArrayList<>();
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));

            Op ada_update = instance.applyGradients(gradsAndVars, "AdGradDATest");

            /**
             * initialize the accumulators
             */
            graph.initializers().forEach((initializer) -> {
                sess.runner().addTarget(initializer).run();
            });

            try (Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> assertEquals(var0_init[index++], f.getFloat(), epsilon));
            }
            try (Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                result.data().scalars().forEach(f -> assertEquals(var1_init[index++], f.getFloat(), epsilon));
            }

            sess.run(ada_update);
            try (Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                float[] expected = {-0.904534F, -1.603567F};
                result.data().scalars().forEach(f -> assertEquals(expected[index++], f.getFloat(), epsilon1));
            }
            try (Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                index = 0;
                float[] expected = {-0.094821F, -0.189358F};
                result.data().scalars().forEach(f -> assertEquals(expected[index++], f.getFloat(), epsilon1));
            }

        }
    }

}
