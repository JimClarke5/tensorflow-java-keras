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
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class CategoricalHingeTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public CategoricalHingeTest() {
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
     * Test of call method, of class CategoricalHinge.
     */
    @Test
    public void testConfig() {
        CategoricalHinge instance = new CategoricalHinge(null);
        assertEquals("categorical_hinge", instance.getName());

        instance = new CategoricalHinge(null, "cat_hinge_loss", Reduction.SUM);
        assertEquals("cat_hinge_loss", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class CategoricalHinge.
     */
    @Test
    public void test_reduction_none() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalHinge instance = new CategoricalHinge(tf, Reduction.NONE);
            int[] true_np = {1, 9, 2, -5};
            float[] pred_np = {4f, 8f, 12f, 8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = instance.call(y_true, y_pred);
            Float[] expected = {0.0f, 65.0f};
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class CategoricalHinge.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5};
            float[] pred_np = {4f, 8f, 12f, 8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 2)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 2)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 32.5f;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class CategoricalHinge.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 83.95f;
            testSession.evaluate(expected, loss);

            Operand loss_2 = instance.call(y_true, y_pred, sampleWeight);
            expected = 83.95f;
            testSession.evaluate(loss, loss_2);
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            float[] weights_np = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(weights_np), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 124.1f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(0f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0f;
            testSession.evaluate(expected, loss);

        }
    }

    @Test
    public void test_timestep_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CategoricalHinge instance = new CategoricalHinge(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            float[] pred_np = {4f, 8f, 12f, 8f, 1f, 3f};
            int[] weights_np = {3, 6, 5, 0, 4, 2};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3, 1)));
            Operand sampleWeight = tf.reshape(tf.constant(weights_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 4.0f;
            testSession.evaluate(expected, loss);

        }
    }

}
