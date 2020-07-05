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

/**
 *
 * @author Jim Clarke
 */
public class HingeTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public HingeTest() {
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
        Hinge instance = new Hinge(null);
        assertEquals("hinge", instance.getName());

        instance = new Hinge(null, "hinge_loss", Reduction.SUM);
        assertEquals("hinge_loss", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class Hinge.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Hinge instance = new Hinge(tf);
            float[] true_np = {0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 0.50625f;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class Hinge.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Hinge instance = new Hinge(tf);
            float[] true_np = {0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 1.164375f;
            testSession.evaluate(expected, loss);

            // todo Verify we get the same output when the same input is given
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Hinge instance = new Hinge(tf);
            float[] sample_narray = {1.2f, 3.4f};
            float[] true_np = {0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 1.06125f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            Hinge instance = new Hinge(tf);
            float[] true_np = {0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f};
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
            Hinge instance = new Hinge(tf, Reduction.AUTO);
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f, 1f, 3f};
            float[] true_np = {0f, 1f, 0f, 1f, 0f, 0f, 1f, 1f};
            float[] pred_np = {-0.3f, 0.2f, -0.1f, 1.6f, -0.25f, -1.f, 0.5f, 0.6f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 4, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 4, 1)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 4)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);

            float expected = 2.0125f;
            testSession.evaluate(expected, loss);
        }
    }

}
