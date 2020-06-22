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
import static org.tensorflow.framework.optimizers.AdaDelta.ACCUMULATOR;
import static org.tensorflow.framework.optimizers.AdaDelta.ACCUMULATOR_UPDATE;
import org.tensorflow.framework.optimizers.Optimizer.GradAndVar;
import static org.tensorflow.keras.optimizers.AdaDelta.EPSILON_DEFAULT;
import static org.tensorflow.keras.optimizers.AdaDelta.EPSILON_KEY;
import static org.tensorflow.keras.optimizers.AdaDelta.LEARNING_RATE_DEFAULT;
import static org.tensorflow.keras.optimizers.AdaDelta.LEARNING_RATE_KEY;
import static org.tensorflow.keras.optimizers.AdaDelta.RHO_DEFAULT;
import static org.tensorflow.keras.optimizers.AdaDelta.RHO_RATE_KEY;
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
public class AdaDeltaTest {

    private int index;

    public AdaDeltaTest() {
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
     * Test of create method, of class AdaDelta.
     */
    @Test
    public void testCreate() {
        System.out.println("create");
        try ( Graph graph = new Graph()) {
            Map<String, Object> config = new HashMap<>();
            config.put(NAME_KEY, "AdaDelta");
            config.put(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
            config.put(RHO_RATE_KEY, RHO_DEFAULT);
            config.put(EPSILON_KEY, EPSILON_DEFAULT);
            AdaDelta expResult = new AdaDelta(graph);
            AdaDelta result = AdaDelta.create(graph, config);
            assertEquals(expResult.getConfig(), result.getConfig());
        }
    }

    @Test
    public void testConstructAdadeltaWithLR() {
        System.out.println("testConstructAdadeltaWithLR");
        try ( Graph graph = new Graph()) {
            AdaDelta opt = new AdaDelta(graph, 1.0F, 0.9F, 1.F);
            AdaDelta opt2 = new AdaDelta(graph, 0.1F, 0.9F, 1.F);
            AdaDelta opt3 = new AdaDelta(graph, 0.1F, 0.9F, 1e-8F);
            String format = "AdaDelta{learningRate=%s, rho=%s, epsilon=%s}";
            String optExpected = String.format(format, 1.0F, 0.9F, 1.F);
            String opt2Expected = String.format(format, 0.1F, 0.9F, 1.F);
            String opt3Expected = String.format(format, 0.1F, 0.9F, 1e-8F);

            String optString = opt.toString();
            String opt2String = opt2.toString();
            String opt3String = opt3.toString();

            assertEquals(optExpected, optString);
            assertEquals(opt2Expected, opt2String);
            assertEquals(opt3Expected, opt3String);
        }

    }

    @Test
    public void testConstructAdadeltaWithEpsilonValues() {
        System.out.println("testConstructAdadeltaWithLR");
        try ( Graph graph = new Graph()) {
            AdaDelta opt = new AdaDelta(graph);
            Map<String, Object> config = opt.getConfig();
            assertEquals(EPSILON_DEFAULT, (float) config.get(EPSILON_KEY));

            opt = new AdaDelta(graph, LEARNING_RATE_DEFAULT, RHO_DEFAULT, 1e-8F);
            config = opt.getConfig();
            assertEquals(1e-8F, (float) config.get(EPSILON_KEY));
        }
    }

    @Test
    public void testBasic() {
        System.out.println("testBasic");
        int num_updates = 4; // # number of ADADELTA steps to perform
        float[] grads = {0.2F, 0.1F, 0.01F};
        float[] lrs = {1.0F, 0.5F, 0.1F};
        for (float grad : grads) {
            for (float lr : lrs) {
                try ( Graph graph = new Graph();  Session sess = new Session(graph)) {
                    Ops tf = Ops.create(graph).withName("test");
                    float[] var0_init = {1.0F, 2.0F};
                    float[] var1_init = {3.0F, 4.0F};
                    float[] fgrads = {grad, grad};
                    Shape shape = Shape.of(var0_init.length);
                    Variable<TFloat32> var0 = tf.withName("var0").variable(shape, TFloat32.DTYPE);
                    Variable<TFloat32> var1 = tf.withName("var1").variable(shape, TFloat32.DTYPE);

                    Assign<TFloat32> var0Initializer = tf.assign(var0, tf.constant(var0_init));
                    Assign<TFloat32> var1Initializer = tf.assign(var1, tf.constant(var1_init));

                    Constant<TFloat32> cgrads = tf.constant(fgrads);

                    float accum = 0.0F;
                    float accum_update = 0.0F;
                    float rho = 0.95F;
                    float epsilon = 1e-8F;
                    float epsilon1 = 1e-5F;

                    /* build the GradsAnvVars */
                    List gradsAndVars = new ArrayList<>();
                    gradsAndVars.add(new GradAndVar<>(cgrads.asOutput(), var0.asOutput()));
                    gradsAndVars.add(new GradAndVar<>(cgrads.asOutput(), var1.asOutput()));

                    /* get the Optimizer */
                    AdaDelta adaDelta = new AdaDelta(graph, lr, rho, epsilon);

                    /**
                     * apply gradients
                     */
                    Op adadelta_update = adaDelta.applyGradients(gradsAndVars, "AdaDeltaTest");

                    /* Create and validae the shapes of the slota */
                    Variable<TFloat32>[] slots = new Variable[2];
                    Variable<TFloat32>[] slotUpdates = new Variable[2];

                    slots[0] = adaDelta.getSlot(var0.asOutput(), ACCUMULATOR).get();
                    assertEquals(slots[0].asOutput().shape(), var0.asOutput().shape());

                    slotUpdates[0] = adaDelta.getSlot(var0.asOutput(), ACCUMULATOR_UPDATE).get();
                    assertEquals(slotUpdates[0].asOutput().shape(), var0.asOutput().shape());

                    slots[1] = adaDelta.getSlot(var1.asOutput(), ACCUMULATOR).get();
                    assertEquals(slots[1].asOutput().shape(), var1.asOutput().shape());

                    slotUpdates[1] = adaDelta.getSlot(var1.asOutput(), ACCUMULATOR_UPDATE).get();
                    assertEquals(slotUpdates[1].asOutput().shape(), var1.asOutput().shape());

                    /* initialize the local variables */
                    sess.runner().addTarget(var0Initializer).run();
                    sess.runner().addTarget(var1Initializer).run();


                    /**
                     * initialize the accumulators
                     */
                    for(Op initializer : graph.initializers()) {
                        sess.runner().addTarget(initializer).run();
                    }

                    /**
                     * make sure the variables were initialized properly
                     */
                    try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        result.data().scalars().forEach(f -> assertEquals(var0_init[index++], f.getFloat(), epsilon));
                    }
                    try ( Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                        index = 0;
                        result.data().scalars().forEach(f -> assertEquals(var1_init[index++], f.getFloat(), epsilon));
                    }

                    float[] updates = new float[num_updates];
                    float tot_update = 0;
                    for (int step = 0; step < num_updates; step++) {
                        sess.run(adadelta_update);
                        accum = accum * rho + (float) Math.pow(grad, 2) * (1.0F - rho);
                        updates[step] = ((float) Math.sqrt(accum_update + epsilon)
                                * (float) (1 / Math.sqrt(accum + epsilon)) * grad);
                        accum_update = (accum_update * rho + ((float) Math.pow(updates[step], 2) * (1.0F - rho)));
                        tot_update += updates[step] * lr;

                        for (int i = 0; i < 2; i++) {
                            final float faccum = accum;
                            try ( Tensor<TFloat32> result = sess.runner().fetch(slots[i]).run().get(0).expect(TFloat32.DTYPE)) {
                                result.data().scalars().forEach(f -> assertEquals(faccum, f.getFloat(), epsilon1));
                            }
                            final float faccum_update = accum_update;
                            try ( Tensor<TFloat32> result = sess.runner().fetch(slotUpdates[i]).run().get(0).expect(TFloat32.DTYPE)) {
                                result.data().scalars().forEach(f -> assertEquals(faccum_update, f.getFloat(), epsilon1));
                            }
                        }

                        final float[] var0_initUpdate = {var0_init[0] - tot_update, var0_init[1] - tot_update};

                        try ( Tensor<TFloat32> result = sess.runner().fetch(var0).run().get(0).expect(TFloat32.DTYPE)) {
                            index = 0;
                            result.data().scalars().forEach(f -> assertEquals(var0_initUpdate[index++], f.getFloat(), epsilon1));
                        }
                        final float[] var1_initUpdate = {var1_init[0] - tot_update, var1_init[1] - tot_update};
                        try ( Tensor<TFloat32> result = sess.runner().fetch(var1).run().get(0).expect(TFloat32.DTYPE)) {
                            index = 0;
                            result.data().scalars().forEach(f -> assertEquals(var1_initUpdate[index++], f.getFloat(), epsilon1));
                        }

                    }

                }
            }
        }
    }

}
