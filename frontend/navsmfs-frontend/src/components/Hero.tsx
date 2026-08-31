import React, { useState, useRef } from 'react';
import { ArrowUpRight } from 'lucide-react';

export default function Hero() {
  const [mousePos, setMousePos] = useState({ x: 50, y: 50 });
  const heroRef = useRef<HTMLElement>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!heroRef.current) return;
    const { left, top, width, height } = heroRef.current.getBoundingClientRect();
    const x = ((e.clientX - left) / width) * 100;
    const y = ((e.clientY - top) / height) * 100;
    setMousePos({ x, y });
  };

  return (
    <section ref={heroRef} id="home" onMouseMove={handleMouseMove} style={{
      position: 'relative', height: '100vh', width: '100%',
      display: 'flex', alignItems: 'center', overflow: 'hidden'
    }}>
      {/* Background Image */}
      <div style={{
        position: 'absolute', inset: -20,
        backgroundImage: 'url(/hero-portrait.jpg)',
        backgroundSize: 'cover', backgroundPosition: 'center',
        filter: 'brightness(0.5)',
        transition: 'transform 0.2s ease-out',
        transform: `translate(${(mousePos.x - 50) * -0.02}%, ${(mousePos.y - 50) * -0.02}%) scale(1.02)`,
        zIndex: 1
      }} />

      {/* Forensic Reveal Layer */}
      <div style={{
        position: 'absolute', inset: -20,
        backgroundImage: 'url(/hero-portrait.jpg)',
        backgroundSize: 'cover', backgroundPosition: 'center',
        filter: 'brightness(1.2) contrast(1.2) sepia(100%) hue-rotate(150deg) saturate(300%)',
        maskImage: `radial-gradient(circle 180px at ${mousePos.x}% ${mousePos.y}%, black, transparent)`,
        WebkitMaskImage: `radial-gradient(circle 180px at ${mousePos.x}% ${mousePos.y}%, black, transparent)`,
        opacity: 0.85,
        pointerEvents: 'none',
        transition: 'mask-position 0.1s ease-out',
        transform: `translate(${(mousePos.x - 50) * -0.02}%, ${(mousePos.y - 50) * -0.02}%) scale(1.02)`,
        zIndex: 2
      }}>
        <div style={{
          position: 'absolute', inset: 0,
          backgroundSize: '4px 4px',
          backgroundImage: 'linear-gradient(rgba(77, 208, 225, 0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(77, 208, 225, 0.3) 1px, transparent 1px)'
        }} />
      </div>

      {/* Grid overlay for cinematic tech feel */}
      <div style={{
        position: 'absolute', inset: 0, zIndex: 3,
        background: 'radial-gradient(circle at center, transparent 0%, var(--color-bg-primary) 100%)',
        opacity: 0.8
      }} />

      <div className="container" style={{ position: 'relative', zIndex: 10, width: '100%' }}>
        <div className="animate-fade-in" style={{ maxWidth: '800px' }}>
          <p className="text-label-small" style={{ marginBottom: '2rem' }}>NAVSMFS / IMAGE FORENSICS</p>
          <h1 className="text-display-huge" style={{ marginBottom: '2rem', display: 'flex', flexDirection: 'column' }}>
            <span>AUTHENTICITY</span>
            <span style={{ color: 'var(--color-text-secondary)' }}>IS NO LONGER</span>
            <span>OBVIOUS.</span>
          </h1>
          
          <p className="text-body-large" style={{ maxWidth: '600px', marginBottom: '3rem' }}>
            NAVSMFS analyzes digital images using facial detection and computer-vision based forensic signals to identify potential manipulation and synthetic content.
          </p>

          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
            <a href="#scan" className="btn-primary">
              SCAN AN IMAGE <ArrowUpRight size={18} />
            </a>
            <a href="#how-it-works" className="btn-secondary">
              HOW IT WORKS
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
