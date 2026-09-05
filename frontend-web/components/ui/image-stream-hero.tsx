"use client";

import { cn } from "cn";
import Image from "next/image";
import * as React from "react";

export interface ImageStreamHeroProps {
  /** Ordered list of image URLs for the corridor columns */
  images: readonly string[];
  /** Content overlaid on the hero */
  children?: React.ReactNode;
  className?: string;
  /** Height of the hero section (Tailwind class). Default: h-[600px] */
  heightClass?: string;
}

/**
 * Two-column scrolling image corridor with overlay content.
 * Left column scrolls upward, right column scrolls downward — creates
 * a sense of ambient motion without distracting from the overlay content.
 *
 * Uses pure CSS keyframe animation (no JS scroll), so it's performant
 * and works with SSR/static rendering.
 */
export function ImageStreamHero({
  images,
  children,
  className,
  heightClass = "h-[600px] md:h-[680px]",
}: ImageStreamHeroProps) {
  // Split images across two columns
  const mid = Math.ceil(images.length / 2);
  const leftImages = images.slice(0, mid);
  const rightImages = images.slice(mid);

  // Duplicate for seamless loop
  const leftLoop = [...leftImages, ...leftImages];
  const rightLoop = [...rightImages, ...rightImages];

  return (
    <div className={cn("relative w-full overflow-hidden", heightClass, className)}>
      {/* Image columns — absolute, behind overlay */}
      <div className="absolute inset-0 flex gap-3 px-3 opacity-30">
        {/* Left column — scrolls up */}
        <div className="flex-1 overflow-hidden">
          <div className="animate-scroll-up flex flex-col gap-3">
            {leftLoop.map((src, i) => (
              <div key={i} className="relative h-64 w-full shrink-0 overflow-hidden rounded-xl">
                <Image
                  src={src}
                  alt=""
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 50vw, 33vw"
                  priority={i < 2}
                />
              </div>
            ))}
          </div>
        </div>
        {/* Right column — scrolls down */}
        <div className="flex-1 overflow-hidden">
          <div className="animate-scroll-down flex flex-col gap-3">
            {rightLoop.map((src, i) => (
              <div key={i} className="relative h-64 w-full shrink-0 overflow-hidden rounded-xl">
                <Image
                  src={src}
                  alt=""
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 50vw, 33vw"
                  priority={i < 2}
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Dark gradient overlay — ensures text is always readable */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/60 via-background/80 to-background" />

      {/* Content */}
      <div className="relative z-10 flex h-full flex-col items-center justify-center px-4 text-center">
        {children}
      </div>
    </div>
  );
}
