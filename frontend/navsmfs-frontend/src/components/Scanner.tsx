import React, { useState, useRef } from 'react';
import { Upload, X } from 'lucide-react';

type Stage =
  | 'UPLOAD'
  | 'DETECTING'
  | 'PROCESSING'
  | 'ANALYZING'
  | 'GENERATING'
  | 'RESULT';

type VerificationResult = {
  label: 'LIKELY AUTHENTIC' | 'LIKELY MANIPULATED';
  score: number;

  facialConsistency: string;
  visualConsistency?: string;
  textureAnomaly?: string;
  forensicAnomalyLevel?: string;

  explanation: string;

  heatmapUrl?: string | null;
  mediaUrl?: string | null;

  facesCount: number;
};

export default function Scanner() {

  const [stage, setStage] =
    useState<Stage>('UPLOAD');

  const [file, setFile] =
    useState<File | null>(null);

  const [preview, setPreview] =
    useState<string | null>(null);

  const [result, setResult] =
    useState<VerificationResult | null>(null);

  const fileInputRef =
    useRef<HTMLInputElement>(null);


  // ============================================================
  // FILE SELECTION
  // ============================================================

  const handleFileChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {

    if (
      e.target.files &&
      e.target.files[0]
    ) {

      const selected =
        e.target.files[0];

      setFile(selected);

      setPreview(
        URL.createObjectURL(selected)
      );

      setResult(null);

      setStage('UPLOAD');
    }
  };


  // ============================================================
  // RESET
  // ============================================================

  const reset = () => {

    setStage('UPLOAD');

    setFile(null);

    setPreview(null);

    setResult(null);

    if (fileInputRef.current) {

      fileInputRef.current.value = '';
    }
  };


  // ============================================================
  // START VERIFICATION
  // ============================================================

  const startVerification = async () => {

    if (!file) {

      alert(
        'Please select an image first.'
      );

      return;
    }


    try {

      // ========================================================
      // STAGE 1
      // ========================================================

      setStage('DETECTING');


      // ========================================================
      // PREPARE FORM DATA
      // ========================================================

      const formData =
        new FormData();

      formData.append(
        'media_file',
        file
      );


      // ========================================================
      // STAGE 2
      // ========================================================

      await new Promise(
        resolve =>
          setTimeout(resolve, 500)
      );

      setStage('PROCESSING');


      // ========================================================
      // STAGE 3
      // ========================================================

      await new Promise(
        resolve =>
          setTimeout(resolve, 500)
      );

      setStage('ANALYZING');


      console.log(
        '================================'
      );

      console.log(
        '📤 SENDING IMAGE TO DJANGO'
      );

      console.log(
        '📁 File:',
        file.name
      );

      console.log(
        '================================'
      );


      // ========================================================
      // DJANGO API
      // ========================================================

      const response =
        await fetch(
          'http://127.0.0.1:8000/api/analyze/',
          {
            method: 'POST',

            body: formData,
          }
        );


      console.log(
        '📥 Django HTTP Status:',
        response.status
      );


      // ========================================================
      // HTTP ERROR
      // ========================================================

      if (!response.ok) {

        const errorText =
          await response.text();

        console.error(
          '❌ Django Error:',
          errorText
        );

        throw new Error(
          `Django returned HTTP ${response.status}`
        );
      }


      // ========================================================
      // GET JSON
      // ========================================================

      const data =
        await response.json();


      console.log(
        '================================'
      );

      console.log(
        '🔥 DJANGO RESPONSE'
      );

      console.log(
        data
      );

      console.log(
        '================================'
      );


      // ========================================================
      // BACKEND SUCCESS CHECK
      // ========================================================

      if (
        data.success !== true
      ) {

        throw new Error(
          data.error ||
          'Django analysis failed.'
        );
      }


      // ========================================================
      // STAGE 4
      // ========================================================

      setStage('GENERATING');


      await new Promise(
        resolve =>
          setTimeout(resolve, 500)
      );


      // ========================================================
      // PREDICTION
      // ========================================================

      const predictionText =
        String(
          data.prediction ||
          'Unknown'
        );


      const predictionLower =
        predictionText.toLowerCase();


      // ========================================================
      // DETERMINE RESULT
      // ========================================================

      const isManipulated =
        predictionLower.includes(
          'fake'
        ) ||
        predictionLower.includes(
          'ai'
        ) ||
        predictionLower.includes(
          'generated'
        ) ||
        predictionLower.includes(
          'manipulated'
        );


      // ========================================================
      // CREATE RESULT
      // ========================================================

      const frontendResult:
        VerificationResult = {

        label:
          isManipulated
            ? 'LIKELY MANIPULATED'
            : 'LIKELY AUTHENTIC',


        score:
          Number(
            data.confidence || 0
          ),


        facesCount:
          Number(
            data.faces_count || 0
          ),


        facialConsistency:
          `${Number(
            data.faces_count || 0
          )} face(s) detected`,


        visualConsistency:
          `${Number(
            data.real_probability || 0
          ).toFixed(2)}% real probability`,


        textureAnomaly:
          `${Number(
            data.fake_probability || 0
          ).toFixed(2)}% AI/fake probability`,


        forensicAnomalyLevel:
          predictionText,


        explanation:
          isManipulated

            ? 'The uploaded image shows forensic characteristics consistent with AI-generated or manipulated content.'

            : 'The uploaded image appears consistent with authentic content based on the forensic analysis.',


        heatmapUrl:
          data.heatmap_url ||
          null,


        mediaUrl:
          data.media_url ||
          null,
      };


      // ========================================================
      // DEBUG
      // ========================================================

      console.log(
        '================================'
      );

      console.log(
        '✅ FRONTEND RESULT'
      );

      console.log(
        frontendResult
      );

      console.log(
        '================================'
      );


      // ========================================================
      // SHOW RESULT
      // ========================================================

      setResult(
        frontendResult
      );

      setStage(
        'RESULT'
      );

    }

    catch (error) {

      console.error(
        '================================'
      );

      console.error(
        '❌ IMAGE ANALYSIS FAILED'
      );

      console.error(
        error
      );

      console.error(
        '================================'
      );


      alert(
        `Analysis failed:\n\n${
          error instanceof Error
            ? error.message
            : String(error)
        }`
      );


      setStage(
        'UPLOAD'
      );
    }
  };


  // ============================================================
  // RENDER
  // ============================================================

  return (

    <section
      id="scan"
      className="section-padding"
    >

      <div className="container">


        {/* ======================================================
            HEADER
        ====================================================== */}

        <div
          style={{
            marginBottom: '4rem'
          }}
        >

          <h2 className="text-display-huge">

            VERIFY

            <br />

            <span
              style={{
                color:
                  'var(--color-text-secondary)'
              }}
            >
              THE IMAGE.
            </span>

          </h2>


          <p
            className="text-body-large"
            style={{
              marginTop: '1rem'
            }}
          >
            Upload an image and let NAVSMFS
            inspect its visual evidence.
          </p>

        </div>


        {/* ======================================================
            MAIN SCANNER
        ====================================================== */}

        <div
          style={{
            backgroundColor:
              'var(--color-bg-secondary)',

            minHeight: '600px',

            position: 'relative',

            display: 'flex',

            flexDirection: 'column'
          }}
        >


          {/* ====================================================
              UPLOAD
          ==================================================== */}

          {stage === 'UPLOAD' && (

            <div
              style={{
                flex: 1,

                display: 'flex',

                flexDirection: 'column',

                alignItems: 'center',

                justifyContent: 'center',

                padding: '4rem'
              }}
            >

              {!preview ? (

                <div
                  onClick={() =>
                    fileInputRef
                      .current
                      ?.click()
                  }

                  style={{
                    border:
                      '1px dashed var(--color-bg-tertiary)',

                    width: '100%',

                    maxWidth: '600px',

                    padding: '4rem 2rem',

                    display: 'flex',

                    flexDirection:
                      'column',

                    alignItems: 'center',

                    cursor: 'pointer',

                    transition:
                      'var(--transition-normal)'
                  }}

                  onMouseOver={(e) => {

                    e.currentTarget.style
                      .borderColor =
                      'var(--color-text-secondary)';
                  }}

                  onMouseOut={(e) => {

                    e.currentTarget.style
                      .borderColor =
                      'var(--color-bg-tertiary)';
                  }}
                >

                  <Upload
                    size={32}
                    style={{
                      color:
                        'var(--color-text-secondary)',

                      marginBottom: '1rem'
                    }}
                  />


                  <p
                    style={{
                      fontFamily:
                        'var(--font-display)',

                      fontSize: '1.5rem',

                      marginBottom: '0.5rem'
                    }}
                  >
                    Drag & Drop
                  </p>


                  <p
                    className="text-label-small"
                  >
                    OR
                  </p>


                  <p
                    style={{
                      marginTop: '0.5rem',

                      color:
                        'var(--color-accent)',

                      fontWeight: 500
                    }}
                  >
                    SELECT IMAGE ↗
                  </p>


                  <p
                    className="text-label-small"
                    style={{
                      marginTop: '2rem'
                    }}
                  >
                    SUPPORTED:
                    JPG, JPEG, PNG, WEBP
                  </p>

                </div>

              ) : (


                /* =================================================
                   PREVIEW
                   ================================================= */

                <div
                  style={{
                    width: '100%',

                    display: 'flex',

                    gap: '4rem',

                    alignItems:
                      'flex-start'
                  }}
                >


                  {/* IMAGE */}

                  <div
                    style={{
                      flex: 1
                    }}
                  >

                    <img
                      src={preview}
                      alt="Preview"

                      style={{
                        width: '100%',

                        height: 'auto',

                        display: 'block',

                        border:
                          '1px solid var(--color-bg-tertiary)'
                      }}
                    />

                  </div>


                  {/* INFORMATION */}

                  <div
                    style={{
                      flex: 1,

                      display: 'flex',

                      flexDirection:
                        'column',

                      gap: '1rem'
                    }}
                  >

                    <div>

                      <p className="text-label-small">
                        FILENAME
                      </p>

                      <p
                        style={{
                          wordBreak:
                            'break-all'
                        }}
                      >
                        {file?.name}
                      </p>

                    </div>


                    <div>

                      <p className="text-label-small">
                        FILE TYPE
                      </p>

                      <p>
                        {file?.type}
                      </p>

                    </div>


                    <div>

                      <p className="text-label-small">
                        FILE SIZE
                      </p>

                      <p>
                        {(
                          (file?.size || 0) /
                          1024 /
                          1024
                        ).toFixed(2)}{' '}
                        MB
                      </p>

                    </div>


                    {/* BUTTONS */}

                    <div
                      style={{
                        marginTop: '3rem',

                        display: 'flex',

                        gap: '1rem',

                        flexWrap: 'wrap'
                      }}
                    >

                      <button
                        className="btn-primary"
                        onClick={
                          startVerification
                        }
                      >
                        START VERIFICATION ↗
                      </button>


                      <button
                        className="btn-secondary"
                        onClick={reset}
                      >

                        <X size={16} />

                        REMOVE IMAGE

                      </button>

                    </div>

                  </div>

                </div>
              )}

            </div>
          )}


          {/* ====================================================
              ANALYSIS
          ==================================================== */}

          {stage !== 'UPLOAD' &&
            stage !== 'RESULT' && (

            <div
              style={{
                flex: 1,

                display: 'flex',

                position: 'relative',

                overflow: 'hidden'
              }}
            >

              {/* IMAGE */}

              <div
                style={{
                  flex: 1,

                  position: 'relative'
                }}
              >

                <img
                  src={preview!}
                  alt="Analysis"

                  style={{
                    width: '100%',

                    height: '100%',

                    objectFit: 'cover',

                    opacity: 0.5,

                    filter:
                      'grayscale(100%)'
                  }}
                />


                <div
                  className="forensic-scan-line"
                  style={{
                    animationDuration:
                      '2s'
                  }}
                />


                {stage === 'DETECTING' && (

                  <div
                    style={{
                      position:
                        'absolute',

                      top: '30%',

                      left: '30%',

                      width: '40%',

                      height: '40%',

                      border:
                        '1px solid var(--color-accent)',

                      boxShadow:
                        '0 0 20px rgba(77,208,225,0.2)'
                    }}
                  />

                )}


                {stage === 'PROCESSING' && (

                  <div
                    style={{
                      position:
                        'absolute',

                      inset: 0,

                      background:
                        'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(77,208,225,0.05) 2px, rgba(77,208,225,0.05) 4px)'
                    }}
                  />

                )}

              </div>


              {/* SIDEBAR */}

              <div
                style={{
                  width: '300px',

                  backgroundColor:
                    'rgba(8,8,8,0.9)',

                  borderLeft:
                    '1px solid var(--color-bg-tertiary)',

                  padding: '2rem',

                  display: 'flex',

                  flexDirection:
                    'column',

                  gap: '2rem'
                }}
              >

                <div
                  style={{
                    opacity:
                      stage === 'DETECTING'
                        ? 1
                        : 0.4
                  }}
                >

                  <p className="text-label-small">
                    STAGE 01
                  </p>

                  <p>
                    DETECTING FACIAL REGION
                  </p>

                </div>


                <div
                  style={{
                    opacity:
                      stage === 'PROCESSING'
                        ? 1
                        : 0.4
                  }}
                >

                  <p className="text-label-small">
                    STAGE 02
                  </p>

                  <p>
                    PROCESSING IMAGE
                  </p>

                </div>


                <div
                  style={{
                    opacity:
                      stage === 'ANALYZING'
                        ? 1
                        : 0.4
                  }}
                >

                  <p className="text-label-small">
                    STAGE 03
                  </p>

                  <p>
                    ANALYZING FORENSIC SIGNALS
                  </p>

                </div>


                <div
                  style={{
                    opacity:
                      stage === 'GENERATING'
                        ? 1
                        : 0.4
                  }}
                >

                  <p className="text-label-small">
                    STAGE 04
                  </p>

                  <p>
                    GENERATING AUTHENTICATION RESULT
                  </p>

                </div>

              </div>

            </div>
          )}


          {/* ====================================================
              RESULT
          ==================================================== */}

          {stage === 'RESULT' &&
            result && (

            <div
              style={{
                flex: 1,

                display: 'flex'
              }}
            >

              {/* =================================================
                  RESULT IMAGE
                  ================================================= */}

              <div
                style={{
                  flex: 1,

                  position: 'relative',

                  minHeight: '600px',

                  backgroundColor:
                    'var(--color-bg-primary)'
                }}
              >

                <img
                  src={
                    result.heatmapUrl ||
                    result.mediaUrl ||
                    preview!
                  }

                  alt="Forensic Analysis"

                  style={{
                    width: '100%',

                    height: '100%',

                    objectFit: 'contain'
                  }}
                />

              </div>


              {/* =================================================
                  RESULT DETAILS
                  ================================================= */}

              <div
                style={{
                  flex: 1,

                  padding: '4rem',

                  display: 'flex',

                  flexDirection:
                    'column',

                  justifyContent:
                    'center'
                }}
              >


                {/* TITLE */}

                <h3
                  className="text-display-huge"

                  style={{
                    color:
                      result.label ===
                      'LIKELY AUTHENTIC'

                        ? 'var(--color-text-primary)'

                        : 'var(--color-danger)',

                    marginBottom:
                      '1rem',

                    whiteSpace:
                      'pre-line'
                  }}
                >

                  {result.label ===
                    'LIKELY AUTHENTIC'

                    ? 'IMAGE\nVERIFIED'

                    : 'MANIPULATION\nDETECTED'}

                </h3>


                {/* CONFIDENCE */}

                <div
                  style={{
                    fontSize: '4rem',

                    fontFamily:
                      'var(--font-display)',

                    color:
                      'var(--color-accent)',

                    marginBottom:
                      '2rem'
                  }}
                >

                  {result.score.toFixed(2)}%

                </div>


                {/* LABEL */}

                <p
                  className="text-label-small"

                  style={{
                    marginBottom:
                      '1rem',

                    color:
                      result.label ===
                      'LIKELY AUTHENTIC'

                        ? 'var(--color-success)'

                        : 'var(--color-danger)'
                  }}
                >

                  {result.label}

                </p>


                {/* =================================================
                    DETAILS
                    ================================================= */}

                <div
                  style={{
                    display: 'grid',

                    gridTemplateColumns:
                      '1fr 1fr',

                    gap: '1rem',

                    margin: '2rem 0',

                    borderTop:
                      '1px solid var(--color-bg-tertiary)',

                    borderBottom:
                      '1px solid var(--color-bg-tertiary)',

                    padding: '2rem 0'
                  }}
                >


                  {/* FACES */}

                  <div>

                    <p className="text-label-small">
                      FACES DETECTED
                    </p>

                    <p>
                      {result.facesCount}
                    </p>

                  </div>


                  {/* FACIAL */}

                  <div>

                    <p className="text-label-small">
                      FACIAL CONSISTENCY
                    </p>

                    <p>
                      {result.facialConsistency}
                    </p>

                  </div>


                  {/* REAL */}

                  <div>

                    <p className="text-label-small">
                      REAL PROBABILITY
                    </p>

                    <p>
                      {result.visualConsistency}
                    </p>

                  </div>


                  {/* FAKE */}

                  <div>

                    <p className="text-label-small">
                      FAKE PROBABILITY
                    </p>

                    <p>
                      {result.textureAnomaly}
                    </p>

                  </div>


                  {/* FORENSIC */}

                  <div>

                    <p className="text-label-small">
                      FORENSIC RESULT
                    </p>

                    <p>
                      {result.forensicAnomalyLevel}
                    </p>

                  </div>

                </div>


                {/* EXPLANATION */}

                <p
                  className="text-body-large"
                  style={{
                    marginBottom:
                      '3rem'
                  }}
                >
                  {result.explanation}
                </p>


                {/* NEW SCAN */}

                <button
                  className="btn-secondary"
                  onClick={reset}
                >
                  NEW SCAN ↗
                </button>

              </div>

            </div>
          )}


          {/* ====================================================
              FILE INPUT
          ==================================================== */}

          <input
            type="file"

            ref={fileInputRef}

            style={{
              display: 'none'
            }}

            accept="
              image/jpeg,
              image/png,
              image/webp
            "

            onChange={
              handleFileChange
            }
          />

        </div>

      </div>

    </section>
  );
}