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
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class BinaryCrossentropyTest {

    private Mode tf_mode = Mode.EAGER;

    public BinaryCrossentropyTest() {
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
        BinaryCrossentropy instance = new BinaryCrossentropy(null);
        assertEquals("binary_crossentropy", instance.getName());

        instance = new BinaryCrossentropy(null, "bce_1", Reduction.SUM);
        assertEquals("bce_1", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void testAllCorrectUnweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(3, 3)));

            Operand<TFloat32> loss = instance.call(y_true, y_true);

            float expected = 0.0f;
            testSession.evaluate(expected, loss);
            // Test with logits.
            float[] logits_np = {
                100.0f, -100.0f, -100.0f,
                -100.0f, 100.0f, -100.0f,
                -100.0f, -100.0f, 100.0f
            };
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(3, 3)));
            instance = new BinaryCrossentropy(tf, true);

            loss = instance.call(y_true, logits);
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1f, 0f, 1f, 0f};
            float[] pred_np = {1f, 1f, 1f, 0f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 3.83331f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] true_np1 = {1f, 0f, 1f, 0f, 1f, 1f};
            float[] logits_np = {
                100.0f, -100.0f, 100.0f,
                100.0f, 100.0f, -100.0f
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            instance = new BinaryCrossentropy(tf, true);
            loss = instance.call(y_true1, logits);
            expected = 33.33333f;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class BinaryCrossentropy.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1f, 0f, 1f, 0f};
            float[] pred_np = {1f, 1f, 1f, 0f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 8.816612f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] true_np1 = {1f, 0f, 1f, 0f, 1f, 1f};
            float[] logits_np = {
                100.0f, -100.0f, 100.0f,
                100.0f, 100.0f, -100.0f
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            instance = new BinaryCrossentropy(tf, true);
            loss = instance.call(y_true1, logits, sampleWeight);
            expected = 76.66667f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            BinaryCrossentropy instance = new BinaryCrossentropy(tf);
            float[] true_np = {1f, 0f, 1f, 0f};
            float[] pred_np = {1f, 1f, 1f, 0f};
            float[] sample_weight_np = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_weight_np), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 4.59997f;
            testSession.evaluate(expected, loss);

            // Test with logits.
            float[] true_np1 = {1f, 0f, 1f, 0f, 1f, 1f};
            float[] logits_np = {
                100.0f, -100.0f, 100.0f,
                100.0f, 100.0f, -100.0f
            };
            int[] weights_np = {4, 3};
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight1 = tf.constant(weights_np);
            instance = new BinaryCrossentropy(tf, true);
            loss = instance.call(y_true1, logits, sampleWeight1);
            expected = 100f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_no_reduction() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();

            // Test with logits.
            float[] true_np1 = {1f, 0f, 1f, 0f, 1f, 1f};
            float[] logits_np = {
                100.0f, -100.0f, 100.0f,
                100.0f, 100.0f, -100.0f
            };
            Operand y_true1 = tf.reshape(tf.constant(true_np1), tf.constant(Shape.of(2, 3)));
            Operand logits = tf.reshape(tf.constant(logits_np), tf.constant(Shape.of(2, 3)));
            BinaryCrossentropy instance = new BinaryCrossentropy(tf, true, 0.0f, Reduction.NONE);
            Operand loss = instance.call(y_true1, logits);
            Float[] expected = {0.f, 66.666664f};
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_label_smoothing() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            float label_smoothing = 0.1f;
            float[] true_array = {1f, 0f, 1f};
            float[] logits_array = {100.0f, -100.0f, -100.0f};
            Operand y_true = tf.reshape(tf.constant(true_array), tf.constant(Shape.of(1, 3)));
            Operand logits = tf.reshape(tf.constant(logits_array), tf.constant(Shape.of(1, 3)));

            BinaryCrossentropy instance = new BinaryCrossentropy(tf, true, label_smoothing);
            Operand loss = instance.call(y_true, logits);
            float expected = (100.0f + 50.0f * label_smoothing) / 3.0f;
            testSession.evaluate(expected, loss);
        } catch (Exception expected) {

        }
    }

}
