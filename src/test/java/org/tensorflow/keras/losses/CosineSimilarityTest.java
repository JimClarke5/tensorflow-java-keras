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
public class CosineSimilarityTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;
    int axis = 1;

    final float[] np_y_true = {
        1f, 9f, 2f,
        -5f, -2f, 6f
    };
    final float[] np_y_pred = {
        4f, 8f, 12f,
        8f, 1f, 3f
    };

    float[] expectedLoss = {0.720488f, -0.3460499f};

    public CosineSimilarityTest() {
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

    private float mean(float[] v) {
        float sum = 0;
        for (int i = 0; i < v.length; i++) {
            sum += v[i];
        }
        return sum / v.length;
    }

    private float[] mul(float[] v, float scalar) {
        float[] result = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = v[i] * scalar;
        }
        return result;
    }

    private float[] mul(float[] v, float[] b) {
        float[] result = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            result[i] = v[i] * b[i];
        }
        return result;
    }

    /**
     * Test of call method, of class CosineSimilarity.
     */
    @Test
    public void testConfig() {
        CosineSimilarity instance = new CosineSimilarity(null);
        assertEquals("cosine_similarity", instance.getName());

        instance = new CosineSimilarity(null, "cos_loss", Reduction.SUM);
        assertEquals("cos_loss", instance.getName());
        assertEquals(Reduction.SUM, instance.getReduction());

    }

    /**
     * Test of call method, of class CosineSimilarity.
     */
    @Test
    public void test_reduction_none() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf, Reduction.NONE);
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            Operand loss = instance.call(y_true, y_pred);
            Float[] expected = {-0.720488f, 0.3460499f};
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class CosineSimilarity.
     */
    @Test
    public void test_unweighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            Operand loss = instance.call(y_true, y_pred);
            float expected = -mean(expectedLoss);
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class CosineSimilarity.
     */
    @Test
    public void test_scalar_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            int[] true_np = {1, 9, 2, -5, -2, 6};
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            Operand sampleWeight = tf.constant(2.3f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = -mean(mul(expectedLoss, 2.3f));
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_sample_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            float[] weights_np = {1.2f, 3.4f};
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            Operand sampleWeight = tf.reshape(tf.constant(weights_np), tf.constant(Shape.of(2, 1)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = -mean(mul(expectedLoss, weights_np));
            testSession.evaluate(expected, loss);
        }
    }

    @Test
    public void test_zero_weighted() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
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
            CosineSimilarity instance = new CosineSimilarity(tf);
            Shape shape = Shape.of(2, 3, 1);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            float[] weights_np = {3, 6, 5, 0, 4, 2};
            Operand sampleWeight = tf.reshape(tf.constant(weights_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = -2.0f;
            testSession.evaluate(expected, loss);

        }
    }

    @Test
    public void test_axis() {
        try (TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf, 1);
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            Operand loss = instance.call(y_true, y_pred);
            float expected = -mean(expectedLoss);
            testSession.evaluate(expected, loss);
        }
    }

}
