/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
public class CosineSimilarityTest {

    private TestSession.Mode tf_mode = TestSession.Mode.EAGER;
    int axis = 1;

    final float[] np_y_true = {
        1F, 9F, 2F,
        -5F, -2F, 6F
    };
    final float[] np_y_pred = {
        4F, 8F, 12F,
        8F, 1F, 3F
    };

    float[] expectedLoss = {0.720488F, -0.3460499F};

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
        System.out.println("testConfig");
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
        System.out.println("test_reduction_none");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf, Reduction.NONE);
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            Operand loss = instance.call(y_true, y_pred);
            Float[] expected = {-0.720488F, 0.3460499F};
            testSession.evaluate(expected, loss);
        }
    }

    /**
     * Test of call method, of class CosineSimilarity.
     */
    @Test
    public void test_unweighted() {
        System.out.println("test_unweighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
        System.out.println("test_scalar_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
        System.out.println("test_sample_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
        System.out.println("test_zero_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            Shape shape = Shape.of(2, 3);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            Operand sampleWeight = tf.constant(0f);
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = 0F;
            testSession.evaluate(expected, loss);

        }
    }

    @Test
    public void test_timestep_weighted() {
        System.out.println("test_timestep_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
            Ops tf = testSession.getTF();
            CosineSimilarity instance = new CosineSimilarity(tf);
            Shape shape = Shape.of(2, 3, 1);
            Operand y_true = tf.reshape(tf.constant(np_y_true), tf.constant(shape));
            Operand y_pred = tf.reshape(tf.constant(np_y_pred), tf.constant(shape));
            float[] weights_np = {3, 6, 5, 0, 4, 2};
            Operand sampleWeight = tf.reshape(tf.constant(weights_np), tf.constant(Shape.of(2, 3)));
            Operand loss = instance.call(y_true, y_pred, sampleWeight);
            float expected = -2.0F;
            testSession.evaluate(expected, loss);

        }
    }

    @Test
    public void test_axis() {
        System.out.println("test_timestep_weighted");
        try ( TestSession testSession = TestSession.createTestSession(tf_mode)) {
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
