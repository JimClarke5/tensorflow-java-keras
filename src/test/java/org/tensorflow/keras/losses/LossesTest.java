/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
import org.tensorflow.op.core.Placeholder;
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
        System.out.println("testGet_Ops_Object");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
        System.out.println("testGet_Ops_Function");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
        System.out.println("testGet_Supplier");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
        System.out.println("testGet_3args");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
        System.out.println("test KLD");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.KLD(tf, y_true, y_pred);
            Float[] expected = {0.017345406F, 1.1748023F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of kld method, of class Losses.
     */
    @Test
    public void testKld() {
        System.out.println("test kld");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.kld(tf, y_true, y_pred);
            Float[] expected = {0.017345406F, 1.1748023F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of kullback_leibler_divergence method, of class Losses.
     */
    @Test
    public void testKullback_leibler_divergence() {
        System.out.println("test kullback_leibler_divergence");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.kullback_leibler_divergence(tf, y_true, y_pred);
            Float[] expected = {0.017345406F, 1.1748023F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MAE method, of class Losses.
     */
    @Test
    public void testMAE() {
        System.out.println("test MAE"); 
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.MAE(tf, y_true, y_pred);
            Float[] expected = {4.6666665F, 6.3333335F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mae method, of class Losses.
     */
    @Test
    public void testMae() {
        System.out.println("test mae");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.mae(tf, y_true, y_pred);
            Float[] expected = {4.6666665F, 6.3333335F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_absolute_error method, of class Losses.
     */
    @Test
    public void testMean_absolute_error() {
        System.out.println("test mean_absolute_error");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.mean_absolute_error(tf, y_true, y_pred);
            Float[] expected = {4.6666665F, 6.3333335F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MAPE method, of class Losses.
     */
    @Test
    public void testMAPE() {
        System.out.println("test MAPE");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.MAPE(tf, y_true, y_pred);
            Float[] expected = {270.37034782F, 153.33333F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mape method, of class Losses.
     */
    @Test
    public void testMape() {
        System.out.println("test mape");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.mape(tf, y_true, y_pred);
            Float[] expected = {270.37034782F, 153.33333F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_absolute_percentage_error method, of class Losses.
     */
    @Test
    public void testMean_absolute_percentage_error() {
        System.out.println("test mean_absolute_percentage_error");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.mean_absolute_percentage_error(tf, y_true, y_pred);
            Float[] expected = {270.37034782F, 153.33333F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MSE method, of class Losses.
     */
    @Test
    public void testMSE() {
        System.out.println("test MSE");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.MSE(tf, y_true, y_pred);
            Float[] expected = {36.666668F, 62.333332F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mse method, of class Losses.
     */
    @Test
    public void testMse() {
        System.out.println("test mse");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.mse(tf, y_true, y_pred);
            Float[] expected = {36.666668F, 62.333332F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_squared_error method, of class Losses.
     */
    @Test
    public void testMean_squared_error() {
        System.out.println("test mean_squared_error");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.mean_squared_error(tf, y_true, y_pred);
            Float[] expected = {36.666668F, 62.333332F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of MSLE method, of class Losses.
     */
    @Test
    public void testMSLE() {
        System.out.println("test MSLE");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.MSLE(tf, y_true, y_pred);
            Float[] expected = {1.00027791F, 1.87380626F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of msle method, of class Losses.
     */
    @Test
    public void testMsle() {
        System.out.println("test msle");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.msle(tf, y_true, y_pred);
            Float[] expected = {1.00027791F, 1.87380626F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of mean_squared_logarithmic_error method, of class Losses.
     */
    @Test
    public void testMean_squared_logarithmic_error() {
        System.out.println("test mean_squared_logarithmic_error");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_array = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] pred_array = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(2,3)));
            Operand y_pred = tf.reshape(tf.constant(pred_array), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.mean_squared_logarithmic_error(tf, y_true, y_pred);
            Float[] expected = {1.00027791F, 1.87380626F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_3args() {
        System.out.println("test testBinary_crossentropy_3args");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = Losses.binary_crossentropy(tf, y_true, y_pred);
            Float[] expected = {7.666619F, 0.F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_4args_1() {
        System.out.println("test testBinary_crossentropy_4args_1");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np1 = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand y_true = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            Operand loss = Losses.binary_crossentropy(tf, y_true, logits,true);
            Float[] expected = {0.F, 66.666664F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_4args_2() {
        System.out.println("testBinary_crossentropy_4args_2");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float labelSmoothing = 2.0F;
            float[] true_np = {1F, 0F, 1F, 0F};
            float[] pred_np = {1F, 1F, 1F, 0F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = Losses.binary_crossentropy(tf, y_true, y_pred,labelSmoothing);
            Float[] expected = {7.666619F, 15.379093F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Losses.
     */
    @Test
    public void testBinary_crossentropy_5args() {
        System.out.println("testBinary_crossentropy_5args");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {1F, 0F, 1F, 0F, 1F, 1F};
            float[] logits_np = {
                100.0F, -100.0F, 100.0F,
                100.0F, 100.0F, -100.0F
            };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            boolean fromLogits = true;
            float labelSmoothing = 2.0F;
            Operand loss = Losses.binary_crossentropy(tf, yTrue, logits, fromLogits, labelSmoothing);
            Float[] expected = {100.000000F, 33.333332F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_3args() {
        System.out.println("testCategorical_crossentropy_3args");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            
            float[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.categorical_crossentropy(tf, y_true, y_pred);
            Float[] expected = {0.105361F, 0.804668F, 0.061875F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_4args_1() {
        System.out.println("testCategorical_crossentropy_4args_1");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
            Float[] expected = {0.001822F, 0.000459F, 0.169846F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_4args_2() {
        System.out.println("testCategorical_crossentropy_4args_2");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float labelSmoothing = 2.0F;
            float[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.categorical_crossentropy(tf, y_true, y_pred, labelSmoothing);
            Float[] expected = {3.959190F, 1.451939F, 5.046643F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testCategorical_crossentropy_5args() {
        System.out.println("testCategorical_crossentropy_5args");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = true;
            float labelSmoothing = 0.0F;
            float[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.categorical_crossentropy(tf, y_true, logits, fromLogits, labelSmoothing);
            Float[] expected = {0.001822F, 0.000459F, 0.169846F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of categorical_hinge method, of class Losses.
     */
    @Test
    public void testCategorical_hinge() {
        System.out.println("test categorical_hinge");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[] true_np = {1, 9, 2, -5};
            float[] pred_np = {4F, 8F, 12F, 8F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,2)));
            Operand y_pred= tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,2)));
            Operand loss = Losses.categorical_hinge(tf, y_true, y_pred);
            Float[] expected = { 0.0F, 65.0F};
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of cosine_similarity method, of class Losses.
     */
    @Test
    public void testCosine_similarity() {
        System.out.println("test cosine_similarity");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {
                1F, 9F, 2F,
                -5F, -2F, 6F
            };
            float[] pred_np = {
                4F, 8F, 12F,
                8F, 1F, 3F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand y_pred= tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.cosine_similarity(tf, y_true, y_pred);
            Float[] expected = { -0.720488F,   0.3460499F };
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of hinge method, of class Losses.
     */
    @Test
    public void testHinge() {
        System.out.println("test hinge");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,4)));
            Operand loss = Losses.hinge(tf, y_true, y_pred);
            Float[] expected = { 0.6F,   0.412500F };
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of huber method, of class Losses.
     */
    @Test
    public void testHuber() {
        System.out.println("test huber");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {.9f, .2f, .2f, .8f, .4f, .6f};
            float[] pred_np = {1.f, 0.f, 1.f, 1.f, 0.f, 0.f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            float delta = 1.0F;
            Operand loss = Losses.huber(tf, yTrue, yPred, delta);
            Float[] expected = { 0.115000F,   0.093333F };
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of logcosh method, of class Losses.
     */
    @Test
    public void testLogcosh() {
        System.out.println("test logcosh");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {1f, 9f, 2f, -5f, -2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.logcosh(tf, yTrue, yPred);
            Float[] expected = { 4.016654F,   5.641836F };
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of poisson method, of class Losses.
     */
    @Test
    public void testPoisson() {
        System.out.println("test poisson");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {1f, 9f, 2f, 5f, 2f, 6f};
            float[] true_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand loss = Losses.poisson(tf, yTrue, yPred);
            Float[] expected = { -4.631855F,   -1.981310F };
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of sparse_categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testSparse_categorical_crossentropy_4args() {
        System.out.println("test sparse_categorical_crossentropy");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = false;
            int[] true_np = { 0, 1, 2};
            float[] pred_np = {
                .9F, .05F, .05F,
                .5F, .89F, .6F,
                .05F, .01F, .94F
            };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = Losses.sparse_categorical_crossentropy(tf, yTrue, yPred, fromLogits);
            Float[] expected = { 0.105361F, 0.804668F,  0.061875F };
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of sparse_categorical_crossentropy method, of class Losses.
     */
    @Test
    public void testSparse_categorical_crossentropy_5args() {
        System.out.println("test sparse_categorical_crossentropy");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = true;
            int axis = 1;
            int[] true_np = { 0, 1, 2};
            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            
            Operand loss = Losses.sparse_categorical_crossentropy(tf, yTrue, logits, fromLogits, axis);
            Float[] expected = { 0.001822F,   0.000459F, 0.169846F };
            testSession.evaluate(expected, loss);

        }
    }

    /**
     * Test of squared_hinge method, of class Losses.
     */
    @Test
    public void testSquared_hinge() {
        System.out.println("test squared_hinge");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,4)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,4)));
            Operand loss = Losses.squared_hinge(tf, yTrue, yPred);
            Float[] expected = { 0.485000F,   0.243125F };
            testSession.evaluate(expected, loss);

        }
    }
    
    @Test 
    public void test_categorical_crossentropy_loss() {
        System.out.println("test_categorical_crossentropy_loss");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Operand<TInt32> target = tf.random.randomUniformInt(
                    tf.constant(Shape.of(5,1)), tf.constant(0), tf.constant(1));
            RandomUniform ru = new RandomUniform(tf);
            Operand<TFloat32> logits = ru.call(tf.constant(Shape.of(5,1)), TFloat32.DTYPE);
            Operand softmaxOutput = tf.nn.softmax(logits);
            Operand output_from_logit = Losses.categorical_crossentropy(tf, target, logits, true);
            Operand outputFromSoftMax =  Losses.categorical_crossentropy(tf, target, softmaxOutput, false);
            testSession.evaluate(output_from_logit, outputFromSoftMax);
        }
    }
    
    
    
    @Test
    public void test_sparse_categorical_crossentropy_loss() {
        System.out.println("test_sparse_categorical_crossentropy_loss");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Operand<TInt32> target = tf.random.randomUniformInt(
                    tf.constant(Shape.of(5,1)), tf.constant(0), tf.constant(1));
            RandomUniform ru = new RandomUniform(tf);
            Operand<TFloat32> logits = ru.call(tf.constant(Shape.of(5,1)), TFloat32.DTYPE);
            Operand softmaxOutput = tf.nn.softmax(logits);
            Operand output_from_logit = Losses.sparse_categorical_crossentropy(tf, target, logits, true);
            Operand outputFromSoftMax =  Losses.sparse_categorical_crossentropy(tf, target, softmaxOutput, false);
            testSession.evaluate(output_from_logit, outputFromSoftMax);
            
        }
    }
    
    @Test
    public void test_binary_crossentropy_loss() {
        System.out.println("test_binary_crossentropy_loss");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Operand<TInt32> target = tf.random.randomUniformInt(
                    tf.constant(Shape.of(5,1)), tf.constant(0), tf.constant(1));
            RandomUniform ru = new RandomUniform(tf);
            Operand<TFloat32> logits = ru.call(tf.constant(Shape.of(5,1)), TFloat32.DTYPE);
            Operand sigmoidOutput = tf.math.sigmoid(logits);
            Operand output_from_logit = Losses.sparse_categorical_crossentropy(tf, target, logits, true);
            Operand outputFromSigmoid =  Losses.sparse_categorical_crossentropy(tf, target, sigmoidOutput, false);
            testSession.evaluate(output_from_logit, outputFromSigmoid);
            
        }
    }

}
