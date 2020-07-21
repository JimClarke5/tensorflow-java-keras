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
import org.tensorflow.framework.optimizers.Optimizer;
import static org.tensorflow.keras.optimizers.Ftrl.INITIAL_ACCUM_VALUE_KEY;
import static org.tensorflow.keras.optimizers.Ftrl.L1STRENGTH_KEY;
import static org.tensorflow.keras.optimizers.Ftrl.L2STRENGTH_KEY;
import static org.tensorflow.keras.optimizers.Ftrl.L2_SHRINKAGE_REGULARIZATION_STRENGTH_KEY;
import static org.tensorflow.keras.optimizers.Ftrl.LEARNING_RATE_KEY;
import static org.tensorflow.keras.optimizers.Ftrl.LEARNING_RATE_POWER_KEY;
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class FtrlTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    int index;

    public FtrlTest() {
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
     * Test of initConfig method, of class Ftrl.
     */
    @Test
    public void testInitConfig() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Map<String, Object> config = new HashMap<>();
            config.put(NAME_KEY, "Ftrl");
            config.put(LEARNING_RATE_KEY, 2.0F);
            config.put(LEARNING_RATE_POWER_KEY, -0.5F);
            config.put(INITIAL_ACCUM_VALUE_KEY, 0.1F);
            config.put(L1STRENGTH_KEY, 0.0F);
            config.put(L2STRENGTH_KEY, 0.0F);
            config.put(L2_SHRINKAGE_REGULARIZATION_STRENGTH_KEY, 0.0F);
            Ftrl expResult = new Ftrl(tf, 2.0F);
            Ftrl result = Ftrl.create(tf, config);
            assertEquals(expResult.getConfig(), result.getConfig());
        }
    }

    /**
     * Test of getOptimizerName method, of class Ftrl.
     */
    @Test
    public void testGetOptimizerName() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            Ftrl instance = new Ftrl(tf);
            String expResult = "Ftrl";
            String result = instance.getOptimizerName();
            assertEquals(expResult, result);
        }
    }

    @Test
    public void testFtrlWithL1_L2_L2Shrinkage() {
        float[] var0_init = {1.0F, 2.0F};
        float[] var1_init = {4.0F, 3.0F};
        float[] grads0_init = {0.1F, 0.2F};
        float[] grads1_init = {0.01F, 0.02F};
        float epsilon = 1e-8F;
        float epsilon1 = 1e-5F;

        int numSteps = 10;

        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();

            Shape shape0 = Shape.of(var0_init.length);
            Shape shape1 = Shape.of(var1_init.length);
            Variable<TFloat32> var0 = tf.withName("var0").variable(shape0, TFloat32.DTYPE);
            Variable<TFloat32> var1 = tf.withName("var1").variable(shape1, TFloat32.DTYPE);

            Assign<TFloat32> var0Initializer = tf.assign(var0, tf.constant(var0_init));
            Assign<TFloat32> var1Initializer = tf.assign(var1, tf.constant(var1_init));

            Constant<TFloat32> grads0 = tf.constant(grads0_init);
            Constant<TFloat32> grads1 = tf.constant(grads1_init);

            

            float learningRate = 3.0F;

            Ftrl instance = new Ftrl(tf, learningRate,
                    -0.5F, // learningRatePower
                    0.1F, // initial_accumulator_value
                    0.001F, // l1_regularization_strength
                    2.0F, // l2_regularization_strength
                    0.1F // l2_shrinkage_regularization_strength
            );

            /* build the GradsAnvVars */
            List gradsAndVars = new ArrayList<>();
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));

            Op ftrl_update = instance.applyGradients(gradsAndVars, "FtrlTest");

            /* initialize the local variables */
            session.run(var0Initializer);
            session.run(var1Initializer);

            /**
             * initialize the accumulators
             */
            session.run(tf.init());

            
            session.evaluate(var0_init, var0);
            session.evaluate(var1_init, var1);

            for (int i = 0; i < numSteps; i++) {
                session.run(ftrl_update);
            }

            float[] expectedVar0 = {-0.22578995F, -0.44345796F};
            session.evaluate(expectedVar0, var0);
            float[] expectedVar1 = {-0.14378493F, -0.13229476F};
             session.evaluate(expectedVar1, var1);

        }
    }

    @Test
    public void testFtrlWithL1() {
        float[] var0_init = {1.0F, 2.0F};
        float[] var1_init = {4.0F, 3.0F};
        float[] grads0_init = {0.1F, 0.2F};
        float[] grads1_init = {0.01F, 0.02F};
        float epsilon = 1e-8F;
        float epsilon1 = 1e-5F;

        int numSteps = 10;

        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();

            Shape shape0 = Shape.of(var0_init.length);
            Shape shape1 = Shape.of(var1_init.length);
            Variable<TFloat32> var0 = tf.withName("var0").variable(shape0, TFloat32.DTYPE);
            Variable<TFloat32> var1 = tf.withName("var1").variable(shape1, TFloat32.DTYPE);

            Assign<TFloat32> var0Initializer = tf.assign(var0, tf.constant(var0_init));
            Assign<TFloat32> var1Initializer = tf.assign(var1, tf.constant(var1_init));

            Constant<TFloat32> grads0 = tf.constant(grads0_init);
            Constant<TFloat32> grads1 = tf.constant(grads1_init);

            float learningRate = 3.0F;

            Ftrl instance = new Ftrl(tf, learningRate,
                    Ftrl.LEARNING_RATE_POWER_DEFAULT, // learningRatePower
                    0.1F, // initial_accumulator_value
                    0.001F, // l1_regularization_strength
                    0.0F, // l2_regularization_strength
                    Ftrl.L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT // l2_shrinkage_regularization_strength
            );

            /* build the GradsAnvVars */
            List gradsAndVars = new ArrayList<>();
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));

            Op ftrl_update = instance.applyGradients(gradsAndVars, "FtrlTest");

            /* initialize the local variables */
            session.run(var0Initializer);
            session.run(var1Initializer);

            /**
             * initialize the accumulators
             */
            session.run(tf.init());
            
            session.evaluate(var0_init, var0);
            session.evaluate(var1_init, var1);

            

            for (int i = 0; i < numSteps; i++) {
                session.run(ftrl_update);
            }

            float[] expectedVar0 = {-7.66718769F, -10.91273689F};
            session.evaluate(expectedVar0, var0);

            float[] expectedVar1 = {-0.93460727F, -1.86147261F};
            session.evaluate(expectedVar1, var1);

        }
    }

    @Test
    public void testFtrlWithL1_L2() {
        float[] var0_init = {1.0F, 2.0F};
        float[] var1_init = {4.0F, 3.0F};
        float[] grads0_init = {0.1F, 0.2F};
        float[] grads1_init = {0.01F, 0.02F};
        float epsilon = 1e-8F;
        float epsilon1 = 1e-5F;

        int numSteps = 10;

        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();

            Shape shape0 = Shape.of(var0_init.length);
            Shape shape1 = Shape.of(var1_init.length);
            Variable<TFloat32> var0 = tf.withName("var0").variable(shape0, TFloat32.DTYPE);
            Variable<TFloat32> var1 = tf.withName("var1").variable(shape1, TFloat32.DTYPE);

            Assign<TFloat32> var0Initializer = tf.assign(var0, tf.constant(var0_init));
            Assign<TFloat32> var1Initializer = tf.assign(var1, tf.constant(var1_init));

            Constant<TFloat32> grads0 = tf.constant(grads0_init);
            Constant<TFloat32> grads1 = tf.constant(grads1_init);


            float learningRate = 3.0F;

            Ftrl instance = new Ftrl(tf, learningRate,
                    Ftrl.LEARNING_RATE_POWER_DEFAULT, // learningRatePower
                    0.1F, // initial_accumulator_value
                    0.001F, // l1_regularization_strength
                    2.0F, // l2_regularization_strength
                    Ftrl.L2_SHRINKAGE_REGULARIZATION_STRENGTH_DEFAULT // l2_shrinkage_regularization_strength
            );

            /* build the GradsAnvVars */
            List gradsAndVars = new ArrayList<>();
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));

            Op ftrl_update = instance.applyGradients(gradsAndVars, "FtrlTest");

             /* initialize the local variables */
            session.run(var0Initializer);
            session.run(var1Initializer);

            /**
             * initialize the accumulators
             */
            session.run(tf.init());
            
            session.evaluate(var0_init, var0);
            session.evaluate(var1_init, var1);

            for (int i = 0; i < numSteps; i++) {
                session.run(ftrl_update);
            }

            float[] expectedVar0 = {-0.24059935F, -0.46829352F};
            session.evaluate(expectedVar0, var0);

            float[] expectedVar1 = {-0.02406147F, -0.04830509F};
            session.evaluate(expectedVar1, var1);

        }
    }

    @Test
    public void doTestFtrlwithoutRegularization() {
        float[] var0_init = {0.0F, 0.0F};
        float[] var1_init = {0.0F, 0.0F};
        float[] grads0_init = {0.1F, 0.2F};
        float[] grads1_init = {0.01F, 0.02F};
        float epsilon = 1e-8F;
        float epsilon1 = 1e-5F;

        int numSteps = 3;

        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();

            Shape shape0 = Shape.of(var0_init.length);
            Shape shape1 = Shape.of(var1_init.length);
            Variable<TFloat32> var0 = tf.withName("var0").variable(shape0, TFloat32.DTYPE);
            Variable<TFloat32> var1 = tf.withName("var1").variable(shape1, TFloat32.DTYPE);

            Assign<TFloat32> var0Initializer = tf.assign(var0, tf.constant(var0_init));
            Assign<TFloat32> var1Initializer = tf.assign(var1, tf.constant(var1_init));

            Constant<TFloat32> grads0 = tf.constant(grads0_init);
            Constant<TFloat32> grads1 = tf.constant(grads1_init);

            float learningRate = 3.0F;

            Ftrl instance = new Ftrl(tf, learningRate);

            /* build the GradsAnvVars */
            List gradsAndVars = new ArrayList<>();
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads0.asOutput(), var0.asOutput()));
            gradsAndVars.add(new Optimizer.GradAndVar<>(grads1.asOutput(), var1.asOutput()));
            Op ftrl_update = instance.applyGradients(gradsAndVars, "FtrlTest");

             /* initialize the local variables */
            session.run(var0Initializer);
            session.run(var1Initializer);

            /**
             * initialize the accumulators
             */
            session.run(tf.init());
            
            session.evaluate(var0_init, var0);
            session.evaluate(var1_init, var1);

            for (int i = 0; i < numSteps; i++) {
                session.run(ftrl_update);
            }

            float[] expectedVar0 = {-2.60260963F, -4.29698515F};
            float[] expectedVar1 = {-0.28432083F, -0.56694895F};

            session.evaluate(expectedVar0, var0);
            session.evaluate(expectedVar1, var1);
        }
    }

}
