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
public class SquaredHingeTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public SquaredHingeTest() {
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
     * Test of call method, of class SquaredHinge.
     */
    @Test
    public void testConfig() {
        SquaredHinge instance = new SquaredHinge(null);
        assertEquals("squared_hinge", instance.getName());

        instance = new SquaredHinge(null, "squared_hinge_loss", Reduction.SUM);
        assertEquals("squared_hinge_loss", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class SquaredHinge.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SquaredHinge instance = new SquaredHinge(tf);
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 0.364062f;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class SquaredHinge.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SquaredHinge instance = new SquaredHinge(tf);
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0.8373437f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SquaredHinge instance = new SquaredHinge(tf);
            float[] sample_narray = {1.2f, 3.4f};
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0.7043125f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SquaredHinge instance = new SquaredHinge(tf);
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand sampleWeight = tf.constant(0.F);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_timestep_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            SquaredHinge instance = new SquaredHinge(tf, Reduction.AUTO);
            float[] true_np = {0, 1, 0, 1, 0, 0, 1, 1};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4, 1)));
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f, 1f, 3f};
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 4)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);

            float expected = 1.54250000f;
            testSession.evaluate(expected, loss);
        }
    }

}
