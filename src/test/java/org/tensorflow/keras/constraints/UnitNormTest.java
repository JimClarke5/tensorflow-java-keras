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
package org.tensorflow.keras.constraints;

import com.google.common.util.concurrent.AtomicDouble;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.keras.utils.ND;
import org.tensorflow.keras.utils.TestSession;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.FloatNdArray;
import org.tensorflow.tools.ndarray.NdArrays;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author jbclarke
 */
public class UnitNormTest {
    private TestSession.Mode tf_mode = TestSession.Mode.GRAPH;
    
    public UnitNormTest() {
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
     * Test of getConfig method, of class Constant.
     */
    @Test
    public void testConfig() {
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            UnitNorm instance = new UnitNorm(tf, 1);
            
            assertEquals(1, instance.getAxis());
        }
    }

    /**
     * This is a uniform random sample from np.rand.random based on the python test for UnitNorm
     */
    private float[] sampleData = {
        0.000000000f, -28.436932210f, -46.849248097f, 20.664176493f, -26.808162355f, -7.227334294f, 37.333857622f, -23.319853830f, 30.349067353f, -22.655558299f,
        13.636556124f, -4.719605707f, -17.009956346f, 45.909142597f, -29.790549866f, 8.163494904f, 38.045532021f, 31.463388013f, -1.281061991f, -37.887445791f,
        10.967848421f, 8.733192011f, 26.844166742f, 25.802080642f, 42.286602053f, 23.789240270f, 0.652000342f, 15.678112202f, -2.340722401f, -43.042927853f,
        -18.818957408f, 12.505786544f, 3.050042934f, 46.017120249f, -48.671799717f, -1.543481481f, -14.069263574f, 9.235298345f, 25.512753258f, 15.390638307f,
        28.800217669f, 17.746018570f, 34.648520997f, -47.343709299f, 48.853396951f, 5.103719940f, 22.035165534f, -9.815911316f, 21.466164053f, 30.532907056f,
        47.423051070f, -7.087221529f, -45.418542866f, 19.917270023f, -32.081996162f, 33.009851459f, 5.486191114f, -25.368051120f, 17.639676213f, 22.820304683f,
        34.968814945f, 37.819129021f, -18.850212173f, -49.535459195f, -45.028089848f, 19.307126238f, -17.547002485f, -7.016414774f, 45.467645424f, -19.235894520f,
        15.087715310f, 13.545817730f, -14.402264766f, -37.671278835f, 21.868669928f, -7.983829614f, 0.241487600f, -45.406377316f, -6.084164290f, 18.363992798f,
        44.912753442f, 10.652103195f, 39.184479874f, -40.694549260f, -20.020929757f, 9.726668210f, -21.587398268f, 43.522551575f, 43.265718466f, 30.889002932f,
        -6.395922079f, 19.322261476f, 13.534218968f, -29.060860575f, -19.815948708f, 0.998496944f, 8.711986801f, -5.548133017f, -28.316970702f, 1.629183653f
    };
    /**
     * Test of call method, of class UnitNorm.
     */
    @Test
    public void testCall() {
        
        try (TestSession session = TestSession.createTestSession(tf_mode)) {
            Ops tf = session.getTF();
            final float[] array = sampleData; 
            Operand<TFloat32> weights = tf.reshape(tf.constant(array), tf.constant(Shape.of(10,10)));
            UnitNorm instance = new UnitNorm(tf);
            Operand result = instance.call(weights);
            try (Tensor<TFloat32> tensor = session.getGraphSession().runner().fetch(result).run().get(0).expect(TFloat32.DTYPE)) {
                float largestDifference = getLargestDifference(tensor);
                assertTrue(Math.abs(largestDifference) < 10e-5f);
            }
        }
    }

    private float getLargestDifference(Tensor<TFloat32> tensor) {
        FloatNdArray tensorArray = NdArrays.ofFloats(tensor.shape());
        tensor.data().copyTo(tensorArray);
        tensorArray = ND.square(tensorArray);
        FloatNdArray normArray = ND.sum(tensorArray, 0);
        FloatNdArray normOfNormalized = ND.sqrt(normArray);
        FloatNdArray difference = ND.sub(normOfNormalized, 1.0f);
        return (float)ND.max(ND.abs(difference));
        
    }
    
}
