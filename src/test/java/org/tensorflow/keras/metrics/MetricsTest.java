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
package org.tensorflow.keras.metrics;

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
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;

/**
 *
 * @author jbclarke
 */
public class MetricsTest {
    
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public MetricsTest() {
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
     * Test of get method, of class Metrics.
     */
    @Test
    public void testGet_Ops_Object() {
         try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            String metricFunction = "kld";
            Metric result = Metrics.get(tf, metricFunction);
            assertNotNull(result);
            assert (result instanceof org.tensorflow.keras.metrics.KLDivergence);

            Class metricClass = org.tensorflow.keras.metrics.Hinge.class;
            result = Metrics.get(tf, org.tensorflow.keras.metrics.Hinge.class);
            assert (result instanceof org.tensorflow.keras.metrics.Hinge);

            Metric instance = new CosineSimilarity(tf);
            result = Metrics.get(tf, instance);
            assertNotNull(result);
            assert (result instanceof CosineSimilarity);
        }
    }

    /**
     * Test of get method, of class Metrics.
     */
    @Test
    public void testGet_Ops_Function() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Function<Ops, Metric> lambda = (ops) -> new LogCoshError(ops);
            Metric expResult = null;
            Metric result = Metrics.get(tf, lambda);
            assertNotNull(result);
            assert (result instanceof LogCoshError);
        }
    }

    /**
     * Test of get method, of class Metrics.
     */
    @Test
    public void testGet_Supplier() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Supplier<Metric> lambda = () -> new org.tensorflow.keras.metrics.MeanSquaredError(tf);
            Metric result = Metrics.get(lambda);
            assertNotNull(result);
            assert (result instanceof org.tensorflow.keras.metrics.MeanSquaredError);
        }
    }

    /**
     * Test of get method, of class Metrics.
     */
    @Test
    public void testGet_3args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            String lossFunction = "logits_scce";
            Function<Ops, Metric> lambda = (ops) -> new SparseCategoricalCrossentropy(ops, true);
            Map<String, Function<Ops, Metric>> custom_functions = new HashMap<>();
            custom_functions.put(lossFunction, lambda);
            Metric result = Metrics.get(tf, lossFunction, custom_functions);
            assertNotNull(result);
            assert (result instanceof SparseCategoricalCrossentropy);
        }
    }

/**
     * Test of KLD method, of class Metrics.
     */
    @Test
    public void testKLD() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.KLD(tf, y_true, y_pred);
            testSession.evaluate(0.5960738F, metric);

        }
    }

    /**
     * Test of kld method, of class Metrics.
     */
    @Test
    public void testKld() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.kld(tf, y_true, y_pred);
            testSession.evaluate(0.5960738F, metric);

        }
    }

    /**
     * Test of kullback_leibler_divergence method, of class Metrics.
     */
    @Test
    public void testKullback_leibler_divergence() {
       try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.kullback_leibler_divergence(tf, y_true, y_pred);
            testSession.evaluate(0.5960738F, metric);

        }
    }
    
    /**
     * Test of MAE method, of class Metrics.
     */
    @Test
    public void testMAE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.MAE(tf, y_true, y_pred);
            testSession.evaluate(0.5f, metric);

        }
    }

    /**
     * Test of mae method, of class Metrics.
     */
    @Test
    public void testMae() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.mae(tf, y_true, y_pred);
            testSession.evaluate(0.5f, metric);

        }
    }

    /**
     * Test of mean_absolute_error method, of class Metrics.
     */
    @Test
    public void testMean_absolute_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.mean_absolute_error(tf, y_true, y_pred);
            testSession.evaluate(0.5f, metric);

        }
    }
    
    /**
     * Test of MAPE method, of class Metrics.
     */
    @Test
    public void testMAPE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.MAPE(tf, y_true, y_pred);
            testSession.evaluate(35e7f, metric);

        }
    }

    /**
     * Test of mape method, of class Metrics.
     */
    @Test
    public void testMape() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.mape(tf, y_true, y_pred);
            testSession.evaluate(35e7f, metric);

        }
    }

    /**
     * Test of mean_absolute_percentage_error method, of class Metrics.
     */
    @Test
    public void testMean_absolute_percentage_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.mean_absolute_percentage_error(tf, y_true, y_pred);
            testSession.evaluate(35e7f, metric);

        }
    }
    
     /**
     * Test of MSE method, of class Metrics.
     */
    @Test
    public void testMSE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.MSE(tf, y_true, y_pred);
            testSession.evaluate(0.5f, metric);

        }
    }

    /**
     * Test of mse method, of class Metrics.
     */
    @Test
    public void testMse() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.mse(tf, y_true, y_pred);
            testSession.evaluate(0.5f, metric);

        }
    }

    /**
     * Test of mean_squared_error method, of class Metrics.
     */
    @Test
    public void testMean_squared_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.mean_squared_error(tf, y_true, y_pred);
            testSession.evaluate(0.5f, metric);

        }
    }
    
    /**
     * Test of MSLE method, of class Metrics.
     */
    @Test
    public void testMSLE() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.MSLE(tf, y_true, y_pred);
            testSession.evaluate(0.24022f, metric);

        }
    }

    /**
     * Test of msle method, of class Metrics.
     */
    @Test
    public void testMsle() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.msle(tf, y_true, y_pred);
            testSession.evaluate(0.24022f, metric);

        }
    }

    /**
     * Test of mean_squared_logarithmic_error method, of class Metrics.
     */
    @Test
    public void testMean_squared_logarithmic_error() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            float[] pred_np = { 
                0, 0, 1, 1, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.mean_squared_logarithmic_error(tf, y_true, y_pred);
            testSession.evaluate(0.24022f, metric);

        }
    }
    
    /**
     * Test of binary_crossentropy method, of class Metrics.
     */
    @Test
    public void testBinary_crossentropy_3args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            int[][] true_np = {{1, 0}, {1, 0}};
            float[][] pred_np = {{1, 1}, {1, 0}};
            Operand y_true = tf.constant(true_np);
            Operand y_pred = tf.constant(pred_np);
            Operand metric = Metrics.binary_crossentropy(tf, y_true, y_pred);
            testSession.evaluate(3.833309f, metric);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Metrics.
     */
    @Test
    public void testBinary_crossentropy_4args_1() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[][] true_np = {{1, 0, 1}, {0, 1, 1}};
            float[][] logits_np = {{100.0F, -100.0F, 100.0F}, {100.0F, 100.0F, -100.0F}};
            Operand y_true = tf.constant(true_np);
            Operand logits = tf.constant(logits_np);
            Operand metric = Metrics.binary_crossentropy(tf, y_true, logits, true);
            testSession.evaluate(33.333332f, metric);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Metrics.
     */
    @Test
    public void testBinary_crossentropy_4args_2() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float labelSmoothing = 0.1f;
            int[] true_np = {1, 0, 1};
            float[] pred_np = {100.f, -100.f, -100.f};
            Operand y_true = tf.constant(true_np);
            Operand y_pred = tf.constant(pred_np);
            Operand metric = Metrics.binary_crossentropy(tf, y_true, y_pred, labelSmoothing);
            testSession.evaluate(5.3972034f, metric);

        }
    }

    /**
     * Test of binary_crossentropy method, of class Metrics.
     */
    @Test
    public void testBinary_crossentropy_5args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[] true_np = {1, 0, 1};
            float[] logits_np = {100.f, -100.f, -100.f};
            Operand yTrue = tf.constant(true_np);
            Operand logits = tf.constant(logits_np);
            boolean fromLogits = true;
            float labelSmoothing = 0.1f;
            Operand metric = Metrics.binary_crossentropy(tf, yTrue, logits, fromLogits, labelSmoothing);
            testSession.evaluate(35f, metric);

        }
    }
    
    
    /**
     * Test of categorical_crossentropy method, of class Metrics.
     */
    @Test
    public void testCategorical_crossentropy_3args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {0.05F, 0.95F, 0F, 0.1F, 0.8F, 0.1F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.categorical_crossentropy(tf, y_true, y_pred);
            testSession.evaluate(1.1769392f, metric);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Metrics.
     */
    @Test
    public void testCategorical_crossentropy_4args_1() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = true;
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] logits_np = {1, 9, 0, 1, 8, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.categorical_crossentropy(tf, y_true, logits, fromLogits);
            testSession.evaluate(3.5011404f, metric);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Metrics.
     */
    @Test
    public void testCategorical_crossentropy_4args_2() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float labelSmoothing = 0.1f;
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {1, 9, 0, 1, 8, 1};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.categorical_crossentropy(tf, y_true, y_pred, labelSmoothing);
            testSession.evaluate(1.4728148f, metric);

        }
    }

    /**
     * Test of categorical_crossentropy method, of class Metrics.
     */
    @Test
    public void testCategorical_crossentropy_5args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = true;
            float labelSmoothing = 0.1f;
            float[][] true_np = {{0, 1, 0}, {0, 0, 1}};
            float[][] logits_np = {{1, 9, 0}, {1, 8, 1}};
            Operand y_true = tf.constant(true_np);
            Operand logits = tf.constant(logits_np);
            Operand metric = Metrics.categorical_crossentropy(tf, y_true, logits, fromLogits, labelSmoothing);
            testSession.evaluate(3.6678069f, metric);

        }
    }
    
    
     /**
     * Test of categorical_hinge method, of class Metrics.
     */
    @Test
    public void testCategorical_hinge() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[] true_np = {
                0, 1, 0, 1, 0,
                0, 0, 1, 1, 1,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 1
            };
            int[] pred_np = {
                0, 0, 1, 1, 0, 
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0,
                1, 1, 1, 1, 1
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(4, 5)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(4, 5)));
            Operand metric = Metrics.categorical_hinge(tf, y_true, y_pred);
            testSession.evaluate(0.5f, metric);

        }
    }

    /**
     * Test of cosine_similarity method, of class Metrics.
     */
    @Test
    public void testCosine_similarity() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[][] true_np = {{ 1, 9, 2}, {-5, -2, 6 }};
            float[][] pred_np = {{ 4, 8, 12}, {8, 1, 3 }};
            Operand y_true = tf.constant(true_np);
            Operand y_pred =tf.constant(pred_np);
            Operand metric = Metrics.cosine_similarity(tf, y_true, y_pred);
            testSession.evaluate(0.18721905f, metric);

        }
    }
    
    

    /**
     * Test of hinge method, of class Metrics.
     */
    @Test
    public void testHinge() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f,
                -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand metric = Metrics.hinge(tf, y_true, y_pred);
            testSession.evaluate(.5062500f, metric);

        }
    }
    
    /**
     * Test of logcosh method, of class Metrics.
     */
    @Test
    public void testLogcoshError() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4, 8, 12, 8, 1, 3};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand metric = Metrics.logCoshError(tf, yTrue, yPred);
            testSession.evaluate(4.829245f, metric);

        }
    }

    /**
     * Test of poisson method, of class Metrics.
     */
    @Test
    public void testPoisson() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float[] true_np = { 4, 8, 12, 8, 1, 3 };
            float[] pred_np = {  1, 9, 2, 5, 2, 6  };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2,3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2,3)));
            Operand metric = Metrics.poisson(tf, yTrue, yPred);
            testSession.evaluate(-3.3065822f, metric);

        }
    }

/**
     * Test of sparse_categorical_crossentropy method, of class Metrics.
     */
    @Test
    public void testSparse_categorical_crossentropy_4args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = false;
            int[] true_np = {1, 2};
            float[] pred_np = {0.05f, 0.95f, 0f, 0.1f, 0.8f, 0.1f};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.sparse_categorical_crossentropy(tf, yTrue, yPred, fromLogits);
            testSession.evaluate(1.1769392f, metric);

        }
    }

    /**
     * Test of sparse_categorical_crossentropy method, of class Metrics.
     */
    @Test
    public void testSparse_categorical_crossentropy_5args() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            boolean fromLogits = false;
            int axis = 0;
            int[] true_np = {1, 2};
            float[][] pred_np = {{0.05f, 0.1f}, {0.95f, 0.8f}, {0f, 0.1f}};
            Operand yTrue = tf.constant(true_np);
            Operand logits = tf.constant(pred_np);

            Operand metric = Metrics.sparse_categorical_crossentropy(tf, yTrue, logits, fromLogits, axis);
            testSession.evaluate(1.1769392f, metric);

        }
    }
    
    /**
     * Test of squared_hinge method, of class Metrics.
     */
    @Test
    public void testSquared_hinge() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
             int[] true_np = { 
                0, 1, 0, 1, 
                0, 0, 1, 1
            };
            float[] pred_np = { 
                -0.3f, 0.2f, -0.1f, 1.6f,
                -0.25f, -1.f, 0.5f, 0.6f
            };
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand metric = Metrics.squared_hinge(tf, yTrue, yPred);
            testSession.evaluate(0.3640625f, metric);

        }
    }

    @Test
    public void test_categorical_crossentropy_loss() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[] true_np = {0, 1, 0, 0, 0, 1};
            float[] pred_np = {0.05F, 0.95F, 0F, 0.1F, 0.8F, 0.1F};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand yPred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            
            Operand metric = Metrics.categorical_crossentropy(tf, yTrue, yPred);
            testSession.evaluate(1.1769392f, metric);
        }
    }

    @Test
    public void test_sparse_categorical_crossentropy_loss() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[] true_np = {1, 2};
            float[] logits_np = {1, 9, 0, 1, 8, 1};
            Operand yTrue = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            Operand metric = Metrics.sparse_categorical_crossentropy(tf, yTrue, logits, true);
            testSession.evaluate(3.501135f, metric);

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
            Operand metric = Metrics.binary_crossentropy(tf, y_true, y_pred);
            testSession.evaluate(3.833309f, metric);

        }
    }

    
    @Test
    public void test_accuracy() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            int[][] true_np = {{2},{1}};
            int[][] pred_np = {{2},{0}};
            
            Operand y_true = tf.constant(new int[][] {{2},{1}});
            Operand y_pred = tf.constant(new int[][]  {{2},{0}});
            Operand sampleWeight = tf.constant(new float[][] {{.5F}, {.2F}});
            Operand metric = Metrics.accuracy(tf, y_true, y_pred, sampleWeight);
            testSession.evaluate(0.71428573f, metric); 
        }
    }
    
    @Test
    public void test_binary_accuracy() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            
            Operand y_true = tf.constant(new int[][] {{1}, {1}});
            Operand y_pred = tf.constant(new int[][] {{1}, {0}});
            Operand sampleWeight = tf.constant(new float[][] {{.5f}, {.2f}});
            Operand metric = Metrics.binary_accuracy(tf, y_true, y_pred, 
                    sampleWeight);
            testSession.evaluate(0.71428573f, metric);
        }
    }
    
    @Test
    public void test_categorical_accuracy() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
        }
    }
    
    @Test
    public void test_top_k_categorical_accuracy() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
        }
    }
    
    @Test
    public void test_sparse_top_k_categorical_accuracy () {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
        }
    }
           
}
