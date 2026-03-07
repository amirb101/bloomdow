import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function riskLevel(elicitationRate: number): { label: string; color: string; bg: string; border: string } {
  if (elicitationRate < 0.2)  return { label: "Low",      color: "text-risk-low",      bg: "bg-risk-low/10",      border: "border-risk-low/30" };
  if (elicitationRate < 0.4)  return { label: "Moderate", color: "text-risk-moderate", bg: "bg-risk-moderate/10", border: "border-risk-moderate/30" };
  if (elicitationRate < 0.65) return { label: "High",     color: "text-risk-high",     bg: "bg-risk-high/10",     border: "border-risk-high/30" };
  return                              { label: "Critical", color: "text-risk-critical", bg: "bg-risk-critical/10", border: "border-risk-critical/30" };
}

export function formatDate(iso: string) {
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

export function modelShortName(model: string) {
  const parts = model.split("/");
  return parts[parts.length - 1];
}
