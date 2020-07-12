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
package org.tensorflow.keras.losses;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.initializers.RandomUniform;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class LossesTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public LossesTest() {
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
     * Test of get method, of class Losses.
     */
    @Test
    public void testGet_Ops_Object() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            String lossFunction = "kld";
            Loss result = Losses.get(tf, lossFunction);
            assertNotNull(result);
            assert (result instanceof KLDivergence);

            Class lossClass = Hinge.class;
            result = Losses.get(tf, Hinge.class);
            assert (result instanceof Hinge);

            Loss instance = new Huber(tf);
            result = Losses.get(tf, instance);
            assertNotNull(result);
            assert (result instanceof Huber);
        }
    }

    /**
     * Test of get method, of class Losses.
     */
    @Test
    public void testGet_Ops_Function() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Function<Ops, Loss> lambda = (ops) -> new LogCosh(ops);
            Loss expResult = null;
            Loss result = Losses.get(tf, lambda);
            assertNotNull(result);
            assert (result instanceof LogCosh);
        }
    }

    /**
     * Test of get method, of class Losses.
     */
    @Test
    public void testGet_Supplier() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Supplier<Loss> lambda = () -> new MeanSquaredError(tf);
            Loss result = Losses.get(lambda);
            assertNotNull(result);
            assert (result instanceof MeanSquaredError);
        }
    }

    /**
     * Test of get method, of class Losses.
     */
    @Test
    public void testGet_3args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            String lossFunction = "logits_scce";
            Function<Ops, Loss> lambda = (ops) -> new SparseCategoricalCrossentropy(ops, true);
            Map<String, Function<Ops, Loss>> custom_functions = new HashMap<>();
            custom_functions.put(lossFunction, lambda);
            Loss result = Losses.get(tf, lossFunction, custom_functions);
            assertNotNull(result);
            assert (result instanceof SparseCategoricalCrossentropy);
        }
    }

    /**
     * Test of KLD method, of class Losses.
     */
    @Test
    public void testKLD() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.KLD(tf, y_true, y_pred);
            Float[] expected = {0.017345406f, 1.1748023f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of kld method, of class Losses.
     */
    @Test
    public void testKld() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.kld(tf, y_true, y_pred);
            Float[] expected = {0.017345406f, 1.1748023f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of kullback_leibler_divergence method, of class Losses.
     */
    @Test
    public void testKullback_leibler_divergence() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.kullback_leibler_divergence(tf, y_true, y_pred);
            Float[] expected = {0.017345406f, 1.1748023f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MAE method, of class Losses.
     */
    @Test
    public void testMAE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.MAE(tf, y_true, y_pred);
            Float[] expected = {4.6666665f, 6.3333335f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mae method, of class Losses.
     */
    @Test
    public void testMae() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.mae(tf, y_true, y_pred);
            Float[] expected = {4.6666665f, 6.3333335f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_absolute_error method, of class Losses.
     */
    @Test
    public void testMean_absolute_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.mean_absolute_error(tf, y_true, y_pred);
            Float[] expected = {4.6666665f, 6.3333335f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MAPE method, of class Losses.
     */
    @Test
    public void testMAPE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.MAPE(tf, y_true, y_pred);
            Float[] expected = {270.37034782f, 153.33333f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mape method, of class Losses.
     */
    @Test
    public void testMape() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.mape(tf, y_true, y_pred);
            Float[] expected = {270.37034782f, 153.33333f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_absolute_percentage_error method, of class Losses.
     */
    @Test
    public void testMean_absolute_percentage_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.mean_absolute_percentage_error(tf, y_true, y_pred);
            Float[] expected = {270.37034782f, 153.33333f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MSE method, of class Losses.
     */
    @Test
    public void testMSE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.MSE(tf, y_true, y_pred);
            Float[] expected = {36.666668f, 62.333332f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mse method, of class Losses.
     */
    @Test
    public void testMse() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.mse(tf, y_true, y_pred);
            Float[] expected = {36.666668f, 62.333332f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_squared_error method, of class Losses.
     */
    @Test
    public void testMean_squared_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.mean_squared_error(tf, y_true, y_pred);
            Float[] expected = {36.666668f, 62.333332f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MSLE method, of class Losses.
     */
    @Test
    public void testMSLE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.MSLE(tf, y_true, y_pred);
            Float[] expected = {1.00027791f, 1.87380626f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of msle method, of class Losses.
     */
    @Test
    public void testMsle() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.msle(tf, y_true, y_pred);
            Float[] expected = {1.00027791f, 1.87380626f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_squared_logarithmic_error method, of class Losses.
     */
    @Test
    public void testMean_squared_logarithmic_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.mean_squared_logarithmic_error(tf, y_true, y_pred);
            Float[] expected = {1.00027791f, 1.87380626f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_3args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            float[] true_np = {1f, 0f, 1f, 0f};
            float[] pred_np = {1f, 1f, 1f, 0f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = Losses.binary_crossentropy(tf, y_true, y_pred);
            Float[] expected = {7.666619f, 0.F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_4args_1() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np1 = {1f, 0f, 1f, 0f, 1f, 1f};
            float[] logits_np = {
                100.0f, -100.0f, 100.0f,
                100.0f, 100.0f, -100.0f
            };
            Operand y_true = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.binary_crossentropy(tf, y_true, logits, true);
            Float[] expected = {0.F, 66.666664f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_4args_2() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float labelSmoothing = 2.0f;
            float[] true_np = {1f, 0f, 1f, 0f};
            float[] pred_np = {1f, 1f, 1f, 0f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = Losses.binary_crossentropy(tf, y_true, y_pred, labelSmoothing);
            Float[] expected = {7.666619f, 15.379093f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_5args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {1f, 0f, 1f, 0f, 1f, 1f};
            float[] logits_np = {
                100.0f, -100.0f, 100.0f,
                100.0f, 100.0f, -100.0f
            };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            boolean fromLogits = true;
            float labelSmoothing = 2.0f;
            Operand loss = Losses.binary_crossentropy(tf, yTrue, logits, fromLogits, labelSmoothing);
            Float[] expected = {100.000000f, 33.333332f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_3args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            float[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.categorical_crossentropy(tf, y_true, y_pred);
            Float[] expected = {0.105361f, 0.804668f, 0.061875f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_4args_1() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = true;
            float[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.categorical_crossentropy(tf, y_true, logits, true);
            Float[] expected = {0.001822f, 0.000459f, 0.169846f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_4args_2() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float labelSmoothing = 2.0f;
            float[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.categorical_crossentropy(tf, y_true, y_pred, labelSmoothing);
            Float[] expected = {3.959190f, 1.451939f, 5.046643f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_5args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = true;
            float labelSmoothing = 0.0f;
            float[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.categorical_crossentropy(tf, y_true, logits, fromLogits, labelSmoothing, -1);
            Float[] expected = {0.001822f, 0.000459f, 0.169846f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_hinge method, of class Losses.
     */
    @Test
    public void testCategorical_hinge() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[] true_np = {1, 9, 2, -5};
            float[] pred_np = {4f, 8f, 12f, 8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = Losses.categorical_hinge(tf, y_true, y_pred);
            Float[] expected = {0.0f, 65.0f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of cosine_similarity method, of class Losses.
     */
    @Test
    public void testCosine_similarity() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {
                1f, 9f, 2f,
                -5f, -2f, 6f
            };
            float[] pred_np = {
                4f, 8f, 12f,
                8f, 1f, 3f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.cosine_similarity(tf, y_true, y_pred);
            Float[] expected = {-0.720488f, 0.3460499f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of hinge method, of class Losses.
     */
    @Test
    public void testHinge() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand loss = Losses.hinge(tf, y_true, y_pred);
            Float[] expected = {0.6f, 0.412500f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of huber method, of class Losses.
     */
    @Test
    public void testHuber() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            float delta = 1.0f;
            Operand loss = Losses.huber(tf, yTrue, yPred, delta);
            Float[] expected = {0.115000f, 0.093333f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of logcosh method, of class Losses.
     */
    @Test
    public void testLogcosh() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.logcosh(tf, yTrue, yPred);
            Float[] expected = {4.016654f, 5.641836f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of poisson method, of class Losses.
     */
    @Test
    public void testPoisson() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {1f, 9f, 2f, 5f, 2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.poisson(tf, yTrue, yPred);
            Float[] expected = {-4.631855f, -1.981310f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of sparse_categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testSparse_categorical_crossentropy_4args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = false;
            int[] true_np = {0, 1, 2};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.sparse_categorical_crossentropy(tf, yTrue, yPred, fromLogits);
            Float[] expected = {0.105361f, 0.804668f, 0.061875f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of sparse_categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testSparse_categorical_crossentropy_5args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = true;
            int axis = 1;
            int[] true_np = {0, 1, 2};
            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));

            Operand loss = Losses.sparse_categorical_crossentropy(tf, yTrue, logits, fromLogits, axis);
            Float[] expected = {0.001822f, 0.000459f, 0.169846f};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of squared_hinge method, of class Losses.
     */
    @Test
    public void testSquared_hinge() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand loss = Losses.squared_hinge(tf, yTrue, yPred);
            Float[] expected = {0.485000f, 0.243125f};
            testSession.evaluate(expected, loss);

        }
    }

    @Test
    public void test_categorical_crossentropy_loss() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Operand<TInt32> target = tf.random.randomUniformInt(
                    tf.constant(Shape.of(5, 1)), tf.constant(0), tf.constant(1));
            RandomUniform ru = new RandomUniform(tf);
            Operand<TFloat32> logits = ru.call(tf.constant(Shape.of(5, 1)), TFloat32.DTYPE);
            Operand softmaxOutput = tf.nn.softmax(logits);
            Operand output_from_logit = Losses.categorical_crossentropy(tf, target, logits, true);
            Operand outputFromSoftMax = Losses.categorical_crossentropy(tf, target, softmaxOutput, false);
            testSession.evaluate(output_from_logit, outputFromSoftMax);
        }
    }

    @Test
    public void test_sparse_categorical_crossentropy_loss() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Operand<TInt32> target = tf.random.randomUniformInt(
                    tf.constant(Shape.of(5, 1)), tf.constant(0), tf.constant(1));
            RandomUniform ru = new RandomUniform(tf);
            Operand<TFloat32> logits = ru.call(tf.constant(Shape.of(5, 1)), TFloat32.DTYPE);
            Operand softmaxOutput = tf.nn.softmax(logits);
            Operand output_from_logit = Losses.sparse_categorical_crossentropy(tf, target, logits, true);
            Operand outputFromSoftMax = Losses.sparse_categorical_crossentropy(tf, target, softmaxOutput, false);
            testSession.evaluate(output_from_logit, outputFromSoftMax);

        }
    }

    @Test
    public void test_binary_crossentropy_loss() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {1f, 0f, 1f, 0f};
            float[] pred_np = {1f, 1f, 1f, 0f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = Losses.binary_crossentropy(tf, y_true, y_pred);
            Float[] expected = {7.6666193f, 0.0f};
            testSession.evaluate(expected, loss);

        }
    }

}
