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
import org.tensorflow.keras.utils.TestSession.Mode;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class CategoricalCrossentropyTest {

    Mode tf_mode = Mode.EAGER;
    float epsilon = 1e-4f;

    public CategoricalCrossentropyTest() {
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

    @Test
    public void testConfig() {
        CategoricalCrossentropy instance = new CategoricalCrossentropy(null);
        assertEquals("categorical_crossentropy", instance.getName());

        instance = new CategoricalCrossentropy(null, "catx_1", Reduction.SUM);
        assertEquals("catx_1", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class CategoricalCrossentropy.
     */
    @Test
    public void testAllCorrectUnweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            long[] true_np = {
                1L, 0L, 0L,
                0L, 1L, 0L,
                0L, 0L, 1L};
            float[] pred_np = {
                1.f, 0.f, 0.f,
                0.f, 1.f, 0.f,
                0.f, 0.f, 1.F};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            Operand<TFloat32> loss = instance.call(y_true, y_pred);
            float expected = 0f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                10.f, 0.f, 0.f,
                0.f, 10.f, 0.f,
                0.f, 0.f, 10.F
            };
            y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            testSession.setEpsilon(1e-3f);
            testSession.evaluate(0.0f, loss);
        }
    }

    /**
     * Test of call method, of class CategoricalCrossentropy.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            int[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 0.32396814f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.f, 1.f, 1.f,
                0.f, 9.f, 1.f,
                2.f, 3.f, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits);
            expected = 0.0573755f;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class CategoricalCrossentropy.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            int[] true_np = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.constant(2.3f);

            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = .7451267f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.f, 1.f, 1.f,
                0.f, 9.f, 1.f,
                2.f, 3.f, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
            loss = instance.call(y_true, logits, sampleWeight);
            expected = 0.13196386f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf);
            float[] sample_weight_np = {1.2f, 3.4f, 5.6f};
            int[] true_np = {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1};
            float[] pred_np = {
                .9f, .05f, .05f,
                .5f, .89f, .6f,
                .05f, .01f, .94f
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(3, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_weight_np), tf.constant(Shape.of(3, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 1.0696f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] logits_np = {
                8.f, 1.f, 1.f,
                0.f, 9.f, 1.f,
                2.f, 3.f, 5.F
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new CategoricalCrossentropy(tf, true);
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
            int[] true_np = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            float[] logits_np = {
                8.f, 1.f, 1.f,
                0.f, 9.f, 1.f,
                2.f, 3.f, 5.F
            };
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, true, 0.0f, Reduction.NONE);
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
            int[] true_array = {1, 0, 0};
            float[] logits_array = {100.0f, -100.0f, -100.0f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(1, 3)));
            Operand logits = tf.reshape(tf.constant(logits_array), tf.constant(Shape.of(1, 3)));

            CategoricalCrossentropy instance = new CategoricalCrossentropy(tf, true, label_smoothing);
            Operand loss = instance.call(y_true, logits);
            float expected = 400.0f * label_smoothing / 3.0f;
            testSession.evaluate(expected, loss);
        } catch (Exception expected) {

        }
    }

}
