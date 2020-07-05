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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class SparseCategoricalCrossentropyTest {

    float epsilon = 1e-4f;
    TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public SparseCategoricalCrossentropyTest() {
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
     * Test of call method, of class SparseSparseCategoricalCrossentropy.
     */
    @Test
    public void testConfig() {
        SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(null);
        assertEquals("sparse_categorical_crossentropy", instance.getName());

        instance = new SparseCategoricalCrossentropy(null, "scc", Reduction.SUM);
        assertEquals("scc", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class SparseCategoricalCrossentropy.
     */
    @Test
    public void testAllCorrectUnweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            testSession.setEpsilon(1e-3f);
            Ops tf = testSession.getTF();

            long[] true_np = {0L, 1L, 2L};
            float[] pred_np = {
                1.F, 0.F, 0.F,
                0.F, 1.F, 0.F,
                0.F, 0.F, 1.F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            Operand<TFloat32> loss = instance.call(y_true, y_pred);
            float expected = 0.0f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                10.F, 0.F, 0.F,
                0.F, 10.F, 0.F,
                0.F, 0.F, 10.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            testSession.evaluate(0.0f, loss);
        }
    }

    /**
     * Test of call method, of class SparseCategoricalCrossentropy.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            int[] true_np = {0, 1, 2};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 0.32396814f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            expected = 0.05737559f;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class SparseCategoricalCrossentropy.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            int[] true_np = {0, 1, 2};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.constant(2.3f);

            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = .7451267f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits, sampleWeight);
            expected = 0.13196386f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf);
            float[] sample_weight_np = {1.2f, 3.4f, 5.6f};
            int[] true_np = {0, 1, 2};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_weight_np), tf.constant(Shape.of(3, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 1.0696f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new SparseCategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits, sampleWeight);
            expected = 0.31829f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_no_reduction() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            // Test with logits.
            int[] true_np = {0, 1, 2};
            float[] logits_np = {
                8.F, 1.F, 1.F,
                0.F, 9.F, 1.F,
                2.F, 3.F, 5.F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, true, 0.0f, Reduction.NONE);
            Operand loss = instance.call(y_true, logits);
            Float[] expected = {0.001822f, 0.000459f, 0.169846f};
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_label_smoothing() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float label_smoothing = 0.1f;
            int[] true_np = {0, 1, 1};
            float[] logits_array = {100.0f, -100.0f, -100.0f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 1)));
            Operand logits = tf.reshape(tf.constant(logits_array), tf.constant(Shape.of(1, 3)));

            SparseCategoricalCrossentropy instance = new SparseCategoricalCrossentropy(tf, true, label_smoothing);
            Operand loss = instance.call(y_true, logits);
            float expected = 400.0f * label_smoothing / 3.0f;
            testSession.evaluate(expected, loss);
        } catch (Exception expected) {

        }
    }

}
