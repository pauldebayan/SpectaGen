const canvas = document.getElementById("canvas");
      const context = canvas.getContext("2d");

      const callSession = async function(){
        window.session = await ort.InferenceSession.create('./generator.onnx');
        console.log("session called");
        document.getElementById("generateButton").disabled = false;
        document.getElementById("generateButton").innerText = "Generate";
      }

      callSession();
      

      let generateNoise = () => {
        let noise = new Float32Array(512).fill(1);
        noise = noise.map(x => x*randn_bm().toExponential(4))
        return new ort.Tensor('float32', Float32Array.from(noise), [1, 512]);
      }
      
      function randn_bm() {
          var u = 0, v = 0;
          while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
          while(v === 0) v = Math.random();
          return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
      }

      function float32ToImageData(floatArray, width, height) {
          // Create Uint8ClampedArray from Float32Array, scaling from [0, 1] to [0, 255]
          const clampedArray = new Uint8ClampedArray(width * height * 4);  // *4 for RGBA

          for (let i = 0; i < width * height; i++) {
              const value = Math.round(floatArray[i]*255);  // Scale to [0, 255]
              clampedArray[i * 4] = value;     // R
              clampedArray[i * 4 + 1] = value; // G
              clampedArray[i * 4 + 2] = value; // B
              clampedArray[i * 4 + 3] = 255;   // Alpha (fully opaque)
          }

          return new ImageData(clampedArray, width, height);
      }


      async function genImage(){

        context.clearRect(0, 0, canvas.width, canvas.height);
      
        const noise =  generateNoise();

        console.log(noise);
         
        const feeds = { "input":  noise};

        const results = await window.session.run(feeds).then(data => {

          let f32a = data[56].cpuData;

          const uint8ClampedArray = new Uint8ClampedArray(512 * 512 * 4);
          for (let i = 0; i < 512; i++) {
            for (let j = 0; j < 512; j++) {
              for (let k = 0; k < 3; k++) {
                const idx = (i * 512 * 4) + (j * 4) + k;
                const f32aIdx = (k * 512 * 512) + (i * 512) + j;
                uint8ClampedArray[idx] = Math.min(255, Math.max(0, f32a[f32aIdx] * 255));
              }
              uint8ClampedArray[(i * 512 * 4) + (j * 4) + 3] = 255;
            }
          }

          const imageData = new ImageData(uint8ClampedArray, canvas.width, canvas.height);

          context.clearRect(0, 0, canvas.width, canvas.height);                              
          context.putImageData(imageData, 0, 0);

        });
        
      }
      