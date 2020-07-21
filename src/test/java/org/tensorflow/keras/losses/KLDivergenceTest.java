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
public class KLDivergenceTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;

    public KLDivergenceTest() {
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
        KLDivergence instance = new KLDivergence(null);
        assertEquals("kullback_leibler_divergence", instance.getName());

        instance = new KLDivergence(null, "kld", Reduction.SUM);
        assertEquals("kld", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class KLDivergence.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            KLDivergence instance = new KLDivergence(tf);
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred);
            float expected = 0.5960738398643668f;
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class KLDivergence.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            KLDivergence instance = new KLDivergence(tf);
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 1.3709698316880434f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            KLDivergence instance = new KLDivergence(tf);
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            float[] sample_narray = {1.2f, 3.4f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 2.0075711736936492f;
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            KLDivergence instance = new KLDivergence(tf);
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3)));
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
            KLDivergence instance = new KLDivergence(tf, Reduction.AUTO);
            float[] pred_np = {.4f, .9f, .12f, .36f, .3f, .4f};
            float[] true_np = {.5f, .8f, .12f, .7f, .43f, .8f};
            float[] sample_narray = {3f, 6f, 5f, 0f, 4f, 2f};
            Operand y_true = tf.reshape(tf.constant(true_np), tf.constant(Shape.of(2, 3, 1)));
            Operand y_pred = tf.reshape(tf.constant(pred_np), tf.constant(Shape.of(2, 3, 1)));
            Operand sampleWeight = tf.reshape(tf.constant(sample_narray), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);

            float expected = 0.2495994912084345f;
            testSession.evaluate(expected, loss);
        }
    }

}
